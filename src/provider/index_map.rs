//! The wire-index → dense-position remap shared by the streaming adapters.

/// Maps a provider's wire block indices onto the agent's flat builder
/// positions. Both adapters drop wire blocks the agent has no builder for
/// (unknown Anthropic block types, skipped Responses reasoning items); a
/// dropped block start would leave every later wire index pointing one
/// builder too far, so surviving blocks are re-keyed onto a dense index
/// space. The flat position of a wire index is its own position in the
/// vector — entries are only ever appended, so entry N is the N-th surviving
/// block and no position needs storing alongside it. A wire index with no
/// entry was dropped, and its deltas are dropped with it.
///
/// Wire indices are assumed unique within a stream — both APIs number blocks
/// monotonically. A repeated index would resolve to its first occurrence.
#[derive(Default)]
pub(crate) struct DenseIndexMap {
    wire_indices: Vec<u32>,
}

impl DenseIndexMap {
    /// Record the surviving block start at `wire_index` and hand out the next
    /// flat builder position.
    pub(crate) fn alloc(&mut self, wire_index: u32) -> usize {
        let index = self.wire_indices.len();
        self.wire_indices.push(wire_index);
        index
    }

    /// The flat builder position of `wire_index`, or `None` if its block
    /// start was dropped.
    pub(crate) fn get(&self, wire_index: u32) -> Option<usize> {
        self.wire_indices.iter().position(|&k| k == wire_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_hands_out_dense_positions_for_sparse_wire_indices() {
        let mut map = DenseIndexMap::default();
        assert_eq!(map.alloc(2), 0);
        assert_eq!(map.alloc(5), 1);
        assert_eq!(map.alloc(9), 2);
    }

    #[test]
    fn get_returns_the_allocated_position_and_none_for_dropped_indices() {
        let mut map = DenseIndexMap::default();
        map.alloc(1);
        map.alloc(3);
        assert_eq!(map.get(1), Some(0));
        assert_eq!(map.get(3), Some(1));
        assert_eq!(map.get(2), None);
    }

    #[test]
    fn a_repeated_wire_index_resolves_to_its_first_occurrence() {
        let mut map = DenseIndexMap::default();
        map.alloc(7);
        map.alloc(7);
        assert_eq!(map.get(7), Some(0));
    }
}
