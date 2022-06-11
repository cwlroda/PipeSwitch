# TODO

1. [x] Redis channel fine-graining between manager and runners
2. [x] Add profiling code to all classes
   - Manager init overhead
   - Pubsub latency
   - Request execution time
3. [x] Extend runner class to send results back to the manager
4. [x] Extend manager class to send results back to clients
5. [ ] Dynamic allocation of GPUs to runners
6. [x] Implement database in redis for fast data lookup (possibility to link up with Scalabel)
7. [ ] Docs and linting
8. [x] Colored logs
9. [ ] Look for potential optimisations in the code (after profiling is set up)

   1. [x] Remove all unnecessary pubsub channels
   2. [ ] Improve model loading performance

10. [ ] Integrate models for more realistic workload testing

    1. [ ] Larger models for throughput and latency testing
    2. [ ] Multi-GPU models for model parallelism testing
11. [x] Convert pubsub to streams for greater message reliability
12. [x] Proper thread and process shutdown (no semaphore leaks)
