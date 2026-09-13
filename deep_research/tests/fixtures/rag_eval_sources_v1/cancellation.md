# Cancellation behavior

取消 research 请求时，必须停止正在运行的 researcher coroutine，确保每个已获取的 concurrency permit 恰好释放一次，并把仍在运行的 task 标记为 cancelled。取消前已完成的 task 保持 completed。Cancellation 不能伪造最终 assistant answer。

# Committed data

Cancellation 不会回滚已经提交的 sources、findings、artifacts、files、messages 或已完成的 task record。部分证据可以保留在线程状态中，但 cancelled task 不能被展示成成功完成。
