// C11 atomics for the recording ring buffer and capture counters. The
// deployment target (macOS 14) predates Swift's Synchronization.Atomic, so
// Swift code reaches lock-free loads and stores through these inline wrappers.
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct RecordingAtomicInt {
    _Atomic(int64_t) value;
} RecordingAtomicInt;

/// Allocates a counter on its own cache line; returns NULL when out of memory.
static inline RecordingAtomicInt *recording_atomic_create(int64_t value) {
    void *memory = NULL;
    if (posix_memalign(&memory, 128, sizeof(RecordingAtomicInt)) != 0) return NULL;
    RecordingAtomicInt *atomic = (RecordingAtomicInt *)memory;
    atomic_init(&atomic->value, value);
    return atomic;
}

static inline void recording_atomic_destroy(RecordingAtomicInt *atomic) { free(atomic); }

static inline int64_t recording_atomic_load_acquire(RecordingAtomicInt *atomic) {
    return atomic_load_explicit(&atomic->value, memory_order_acquire);
}

static inline int64_t recording_atomic_load_relaxed(RecordingAtomicInt *atomic) {
    return atomic_load_explicit(&atomic->value, memory_order_relaxed);
}

static inline void recording_atomic_store_release(RecordingAtomicInt *atomic, int64_t value) {
    atomic_store_explicit(&atomic->value, value, memory_order_release);
}

static inline void recording_atomic_store_relaxed(RecordingAtomicInt *atomic, int64_t value) {
    atomic_store_explicit(&atomic->value, value, memory_order_relaxed);
}

static inline int64_t recording_atomic_fetch_add_relaxed(RecordingAtomicInt *atomic, int64_t delta) {
    return atomic_fetch_add_explicit(&atomic->value, delta, memory_order_relaxed);
}
