# Nvidia 4

Fri, 10 Apr 26

### Interview Overview

- Coding interview with Jafa from Nvidia
- Technical assessment focused on memory allocator design
- Python implementation chosen for solution

### Problem Statement

- Design memory allocator for fixed memory block (100 bytes example)
- Required APIs:
  1. Initialize - set up memory pool with given capacity
  2. Allocate - takes size parameter, returns pointer/index to allocated memory
  3. Free - takes pointer/index, releases memory back to pool
- Memory should be reusable after free operations
- Support variable allocation sizes (10 bytes, 20 bytes, etc.)

### Solution Approach

- Memory pool implementation using index-based tracking
- Two internal data structures:
  - Free map - tracks available memory slots
  - Allocation map - tracks currently allocated memory blocks
- Allocate function:
  - Returns memory buffer index as pointer
  - Searches free slots for adequate space
  - Returns -1 if no suitable space found
- Free function:
  - Requires index and size parameters from caller
  - Adds freed space back to available pool

### Technical Challenges Discussed

- Initial complexity with page-based allocation rejected for simpler approach
- Fragmentation issues identified but not fully resolved
- Memory consolidation/merging of adjacent free blocks:
  - Sort free slots by position
  - Merge adjacent slots into larger blocks
  - Recreate free map with consolidated slots
- API safety concerns:
  - Caller must track allocated index and size
  - Potential for memory corruption if incorrect parameters provided
  - Validation logic needed but not implemented

### Interview Outcome

- Working solution demonstrated with basic functionality
- Fragmentation handling discussed but not fully implemented
- Time constraints prevented complete merge functionality
- Final interview in Ryan’s process with Nvidia

---

Chat with meeting transcript: https://notes.granola.ai/t/915f614c-2870-4f01-aab3-1641cdc745a1-00demib2