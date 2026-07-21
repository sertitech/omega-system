//! Process-wide fan-out concurrency cap, shared by every agent drawing from it.
//!
//! The tool fan-out in [`crate::agent::Agent::execute_tools`] spawns one scoped
//! worker per approved read-only call, and a `task` child spawns its own leaf
//! fan-out in turn — so an unbounded batch (or a tree of them) could grow the
//! process to N + N·M live threads and run that many tool executions at once.
//! [`Concurrency`] is a counting semaphore that bounds both: a worker holds a
//! permit for the whole of its execution, and permits are acquired *before* the
//! worker is spawned, so the ceiling caps live workers and running executions
//! together. Cloning the handle shares the underlying permits (`std::sync` is
//! the whole mechanism, no new crate), which is what makes it one *process-wide*
//! pool — a parent's fan-out and every child's fan-out draw from the same count.

use std::sync::{Arc, Condvar, Mutex};

/// The number of leaf tool executions that may run concurrently across the whole
/// process. A constant, not a config knob (revisit with evidence, as with the
/// budget tier limits): 8 keeps a wide fan-out genuinely parallel while bounding
/// the thread and execution growth a deep `task` tree would otherwise cause.
pub const FANOUT_PERMITS: usize = 8;

/// A shared, thread-safe counting semaphore for the tool fan-out.
///
/// The top-level agent creates one with [`Concurrency::new`]; the `task` tool
/// clones it into every child agent it spawns (via
/// [`crate::agent::Agent::set_concurrency`]), so parent and children contend for
/// the same permits. [`Concurrency::acquire`] blocks while permits are exhausted
/// and returns an RAII [`Permit`] that releases on drop — including on a worker
/// panic, since drop runs during unwinding.
#[derive(Clone)]
pub struct Concurrency(Arc<(Mutex<usize>, Condvar)>);

impl Concurrency {
    /// A pool seeded with the default [`FANOUT_PERMITS`] permits.
    pub fn new() -> Self {
        Self::with_permits(FANOUT_PERMITS)
    }

    /// A pool seeded with `permits` permits. `new` uses it with the default
    /// count; tests use it to shrink the ceiling to a value they can rendezvous
    /// against.
    pub(crate) fn with_permits(permits: usize) -> Self {
        Self(Arc::new((Mutex::new(permits), Condvar::new())))
    }

    /// Take one permit, blocking while none are available, and return a guard
    /// that gives it back on drop. Contending threads wake on each release and
    /// re-check under the lock, so a permit is never handed to two callers.
    pub fn acquire(&self) -> Permit {
        let (lock, cvar) = &*self.0;
        let mut available = lock.lock().unwrap();
        while *available == 0 {
            available = cvar.wait(available).unwrap();
        }
        *available -= 1;
        Permit(Arc::clone(&self.0))
    }
}

impl Default for Concurrency {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl Concurrency {
    /// The number of permits currently free — test-only instrumentation.
    pub(crate) fn available(&self) -> usize {
        *self.0.0.lock().unwrap()
    }
}

/// An acquired fan-out permit. Holds a clone of the pool handle (not a borrow)
/// so it can move into a scoped worker thread and outlive the `acquire` call.
/// Dropping it returns the permit and wakes one waiter — the release happens in
/// [`Drop`], so it runs whether the worker finishes normally or panics.
pub struct Permit(Arc<(Mutex<usize>, Condvar)>);

impl Drop for Permit {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.0;
        *lock.lock().unwrap() += 1;
        cvar.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_with_the_default_permit_count() {
        assert_eq!(Concurrency::new().available(), FANOUT_PERMITS);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Concurrency::default().available(), FANOUT_PERMITS);
    }

    #[test]
    fn permits_are_tracked_as_they_are_acquired_and_released() {
        let pool = Concurrency::with_permits(2);
        assert_eq!(pool.available(), 2);
        let a = pool.acquire();
        assert_eq!(pool.available(), 1);
        let b = pool.acquire();
        assert_eq!(pool.available(), 0);
        drop(b);
        assert_eq!(pool.available(), 1);
        drop(a);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn clone_shares_the_same_permit_pool() {
        let pool = Concurrency::with_permits(1);
        let clone = pool.clone();
        let _held = pool.acquire();
        assert_eq!(clone.available(), 0, "the clone draws from the same pool");
    }

    #[test]
    fn acquire_blocks_until_a_permit_is_released() {
        use std::sync::mpsc;
        use std::time::Duration;

        // The pool is exhausted, so a second acquire must park in `cvar.wait`
        // until the held permit is released. A handshake (`ready`) pins that the
        // waiter has reached the acquire before we assert it is blocked; the
        // bounded negative wait proves it did not slip a permit, and the release
        // proves it then wakes.
        let pool = Concurrency::with_permits(1);
        let held = pool.acquire();
        let pool2 = pool.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            let _permit = pool2.acquire();
            done_tx.send(()).unwrap();
        });

        ready_rx.recv().unwrap();
        assert!(
            done_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "acquire must block while the pool is exhausted"
        );
        drop(held);
        done_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn a_permit_is_released_even_when_the_holder_panics() {
        // The RAII release lives in Drop, which runs during unwinding — a
        // panicking worker must not leak its permit and starve the pool.
        let pool = Concurrency::with_permits(1);
        let pool2 = pool.clone();
        let handle = std::thread::spawn(move || {
            let _permit = pool2.acquire();
            panic!("worker exploded");
        });
        assert!(handle.join().is_err(), "the worker panicked");
        assert_eq!(pool.available(), 1, "the permit was released on unwind");
        // And the freed permit is genuinely reusable — a leak would block here.
        let _permit = pool.acquire();
        assert_eq!(pool.available(), 0);
    }
}
