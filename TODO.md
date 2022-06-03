# TODO

1. [ ] Add profiling code to all classes
   - Manager init overhead
   - Pubsub latency
   - Request execution time
2. [ ] Create results aggregator class
3. [ ] Extend runner class to send results back to the manager
4. [ ] Extend manager class to send results back to clients
5. [ ] Implement database in redis for fast data lookup (possibility to link up with Scalabel)
6. [ ] Docs and linting
7. [ ] Compare mp.Pipe and redis speed to see if using pipes for communication between manager and runners is faster
8. [ ] Look for potential optimisations in the code
9. [ ] Authentication security?
