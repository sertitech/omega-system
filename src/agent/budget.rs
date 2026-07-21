//! Process-wide tool-call budget, shared by every agent drawing from it.
//!
//! A single-process agent used to own its budget outright (`HashMap`s on
//! `Agent`). Child agents spawned by the `task` tool must
//! draw from the *same* per-tier ceiling as their parent — two agents each
//! enforcing their own private limit would jointly admit twice the intended
//! budget. [`BudgetLedger`] is the shared handle: cloning it shares the
//! underlying counters, so every clone observes every other clone's draws.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Default per-process call limits by cost tier, mirroring the cost scale in
/// [`crate::tools::ToolDef::cost`]. Tier 0 (free) and tiers 1-2 (local) are
/// unlimited; tiers 3 (`web_fetch`) and 4 (`web_search`) are capped so a
/// runaway loop can't burn an unbounded number of external calls.
fn default_tier_limits() -> HashMap<u8, u32> {
    let mut limits = HashMap::new();
    limits.insert(3, 20);
    limits.insert(4, 10);
    limits
}

/// The counters and ceilings a [`BudgetLedger`] guards. Kept as one struct,
/// behind one lock, so a draw's check and its increment read and write both
/// under a single acquisition — see [`BudgetLedger::draw`].
struct Budget {
    counts: HashMap<u8, u32>,
    limits: HashMap<u8, u32>,
}

/// A shared, thread-safe ledger of tool-call budgets by cost tier.
///
/// The top-level agent creates one with [`BudgetLedger::new`]; a child agent
/// spawned by the `task` tool receives a clone via
/// [`crate::agent::Agent::set_budget_ledger`]. Cloning an `Arc` shares state
/// rather than forking it, so parent and child draw against the same
/// counters — `std::sync` is the whole mechanism, no new crate needed.
#[derive(Clone)]
pub struct BudgetLedger(Arc<Mutex<Budget>>);

impl BudgetLedger {
    /// A fresh ledger seeded with the default per-tier limits and every
    /// count at zero.
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(Budget {
            counts: HashMap::new(),
            limits: default_tier_limits(),
        })))
    }

    /// Atomically check `cost`'s count against its limit and, if under,
    /// reserve the call by bumping the count — one lock acquisition covers
    /// both the read and the write, so two threads racing to draw from the
    /// same tier on a shared ledger can never both observe room under the
    /// limit: the second sees the first's increment and is rejected. This is
    /// the TOCTOU a separate check-then-increment would leave open once more
    /// than one agent draws from the same ledger.
    ///
    /// A tier absent from the limit table is unlimited and always draws
    /// `Ok`. `Err` carries the observed `(count, limit)` for the caller's
    /// rejection message; the reservation is *not* made in that case.
    ///
    /// The reservation happens before a tool's own validation and
    /// confirmation gates run, so a call already over budget is rejected
    /// before either — matching the pre-shared-ledger ordering. A call that
    /// reserves and then fails one of those gates must call
    /// [`BudgetLedger::release`] to give the slot back; a call that runs
    /// (whether it then succeeds or fails) keeps the reservation, so retries
    /// still count.
    pub fn draw(&self, cost: u8) -> Result<(), (u32, u32)> {
        let mut budget = self.0.lock().unwrap();
        if let Some(&limit) = budget.limits.get(&cost) {
            let count = budget.counts.get(&cost).copied().unwrap_or(0);
            if count >= limit {
                return Err((count, limit));
            }
        }
        *budget.counts.entry(cost).or_insert(0) += 1;
        Ok(())
    }

    /// Undo a [`BudgetLedger::draw`] reservation for a call that was
    /// rejected before it ran (failed validation, denied confirmation). The
    /// budget only bounds calls that actually execute, so a reservation that
    /// never turns into a run must not linger and starve a later, valid
    /// call. Saturating: releasing past zero (which a correct caller never
    /// does) clamps rather than underflows.
    pub fn release(&self, cost: u8) {
        let mut budget = self.0.lock().unwrap();
        if let Some(count) = budget.counts.get_mut(&cost) {
            *count = count.saturating_sub(1);
        }
    }
}

impl Default for BudgetLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl BudgetLedger {
    /// Test-only override of a tier's limit, replacing the old
    /// `agent.tier_limits.insert(cost, limit)` field poke.
    pub(crate) fn set_limit(&self, cost: u8, limit: u32) {
        self.0.lock().unwrap().limits.insert(cost, limit);
    }

    /// Test-only override of a tier's count, replacing the old
    /// `agent.tier_counts.insert(cost, count)` field poke.
    pub(crate) fn set_count(&self, cost: u8, count: u32) {
        self.0.lock().unwrap().counts.insert(cost, count);
    }

    /// Test-only read of a tier's count, replacing the old
    /// `agent.tier_counts.get(&cost)` field poke.
    pub(crate) fn count(&self, cost: u8) -> u32 {
        self.0
            .lock()
            .unwrap()
            .counts
            .get(&cost)
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_admits_calls_under_the_limit() {
        let ledger = BudgetLedger::new();
        ledger.set_limit(4, 2);

        assert!(ledger.draw(4).is_ok());
        assert!(ledger.draw(4).is_ok());
        assert_eq!(ledger.count(4), 2);
    }

    #[test]
    fn draw_rejects_once_the_limit_is_reached() {
        let ledger = BudgetLedger::new();
        ledger.set_limit(4, 1);

        assert!(ledger.draw(4).is_ok());
        assert_eq!(ledger.draw(4), Err((1, 1)));
        // A rejected draw does not reserve — the count stays at the limit,
        // not past it.
        assert_eq!(ledger.count(4), 1);
    }

    #[test]
    fn draw_on_an_unlimited_tier_never_rejects() {
        let ledger = BudgetLedger::new();
        // Tier 1 carries no default limit.
        for _ in 0..1000 {
            assert!(ledger.draw(1).is_ok());
        }
        assert_eq!(ledger.count(1), 1000);
    }

    #[test]
    fn release_gives_a_reservation_back() {
        let ledger = BudgetLedger::new();
        ledger.set_limit(4, 1);

        assert!(ledger.draw(4).is_ok());
        ledger.release(4);
        // The freed slot admits a fresh call.
        assert!(ledger.draw(4).is_ok());
        assert_eq!(ledger.count(4), 1);
    }

    #[test]
    fn release_on_an_empty_tier_saturates_instead_of_underflowing() {
        let ledger = BudgetLedger::new();
        ledger.release(4);
        assert_eq!(ledger.count(4), 0);
    }

    #[test]
    fn clone_shares_state_with_the_original() {
        let ledger = BudgetLedger::new();
        let clone = ledger.clone();

        clone.set_limit(4, 5);
        assert!(ledger.draw(4).is_ok());

        assert_eq!(clone.count(4), 1, "clone must observe the original's draw");
    }

    #[test]
    fn default_matches_new() {
        let ledger = BudgetLedger::default();
        assert_eq!(ledger.count(4), 0);
    }

    #[test]
    fn concurrent_draws_never_admit_past_the_limit() {
        // A rendezvous (mirrors `execute_tools_runs_approved_calls_
        // concurrently_in_block_order`'s pattern in agent/mod.rs) pins
        // genuine concurrency without timing assumptions: every worker
        // blocks on the barrier until all are released together, so their
        // `draw` calls are guaranteed to contend for the same slots instead
        // of running sequentially by scheduling accident. If `draw` were
        // ever split back into a separate check-then-increment, this test
        // would catch admissions past the limit; a single locked operation
        // cannot.
        use std::sync::Barrier;

        const WORKERS: usize = 32;
        const LIMIT: u32 = 5;

        let ledger = BudgetLedger::new();
        ledger.set_limit(4, LIMIT);
        let barrier = Barrier::new(WORKERS);

        let admitted: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WORKERS)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        ledger.draw(4).is_ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .filter(|&ok| ok)
                .count()
        });

        assert_eq!(admitted, LIMIT as usize);
        assert_eq!(ledger.count(4), LIMIT);
    }
}
