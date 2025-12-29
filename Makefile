# 定义日志文件路径

SERVER_LOG := benchmark/multi-model/server.log
CONTROLLER_LOG := benchmark/multi-model/server.log.global_controller.log
GPU_SCHEDULER_LOG := benchmark/multi-model/server.log.gpu_scheduler.log
MODEL_SCHEDULER_LOG := benchmark/multi-model/server.log.model_scheduler.log
BENCHMARK_RESULT := benchmark/multi-model/benchmark-results
BENCHMARK_STATS := benchmark/multi-model/benchmark/multi-model
REQUEST_OUTPUT := benchmark/multi-model/output-requests
# 默认目标：显示帮助信息
.DEFAULT_GOAL := help

# 清理日志文件
clean:
	@if [ -f "$(SERVER_LOG)" ]; then \
		rm -f "$(SERVER_LOG)"; \
		echo "已清理服务器日志: $(SERVER_LOG)"; \
	else \
		echo "服务器日志不存在: $(SERVER_LOG)"; \
	fi
	@if [ -f "$(CONTROLLER_LOG)" ]; then \
		rm -f "$(CONTROLLER_LOG)"; \
		echo "已清理控制器日志: $(CONTROLLER_LOG)"; \
	else \
		echo "控制器日志不存在: $(CONTROLLER_LOG)"; \
	fi
	@if [ -f "$(GPU_SCHEDULER_LOG)" ]; then \
		rm -f "$(GPU_SCHEDULER_LOG)"; \
		echo "已清理GPU调度器日志: $(GPU_SCHEDULER_LOG)"; \
	else \
		echo "GPU调度器日志不存在: $(GPU_SCHEDULER_LOG)"; \
	fi
	@if [ -f "$(MODEL_SCHEDULER_LOG)" ]; then \
		rm -f "$(MODEL_SCHEDULER_LOG)"; \
		echo "已清理模型调度器日志: $(MODEL_SCHEDULER_LOG)"; \
	else \
		echo "模型调度器日志不存在: $(MODEL_SCHEDULER_LOG)"; \
	fi	
	@if [ -d "$(BENCHMARK_RESULT)" ]; then \
		rm -rf "$(BENCHMARK_RESULT)"; \
		echo "已清理测试记录: $(BENCHMARK_RESULT)"; \
	else \
		echo "测试记录不存在: $(BENCHMARK_RESULT)"; \
	fi
	@if [ -d "$(BENCHMARK_STATS)" ]; then \
		rm -rf "$(BENCHMARK_STATS)"; \
		echo "已清理测试统计: $(BENCHMARK_STATS)"; \
	else \
		echo "测试统计不存在: $(BENCHMARK_STATS)"; \
	fi
	@if [ -d "$(REQUEST_OUTPUT)" ]; then \
		rm -rf "$(REQUEST_OUTPUT)"; \
		echo "已清理请求输出: $(REQUEST_OUTPUT)"; \
	else \
		echo "请求输出不存在: $(REQUEST_OUTPUT)"; \
	fi

clean-log:
	@if [ -f "$(SERVER_LOG)" ]; then \
		rm -f "$(SERVER_LOG)"; \
		echo "已清理服务器日志: $(SERVER_LOG)"; \
	else \
		echo "服务器日志不存在: $(SERVER_LOG)"; \
	fi
	@if [ -f "$(CONTROLLER_LOG)" ]; then \
		rm -f "$(CONTROLLER_LOG)"; \
		echo "已清理控制器日志: $(CONTROLLER_LOG)"; \
	else \
		echo "控制器日志不存在: $(CONTROLLER_LOG)"; \
	fi
	@if [ -f "$(GPU_SCHEDULER_LOG)" ]; then \
		rm -f "$(GPU_SCHEDULER_LOG)"; \
		echo "已清理GPU调度器日志: $(GPU_SCHEDULER_LOG)"; \
	else \
		echo "GPU调度器日志不存在: $(GPU_SCHEDULER_LOG)"; \
	fi
	@if [ -f "$(MODEL_SCHEDULER_LOG)" ]; then \
		rm -f "$(MODEL_SCHEDULER_LOG)"; \
		echo "已清理模型调度器日志: $(MODEL_SCHEDULER_LOG)"; \
	else \
		echo "模型调度器日志不存在: $(MODEL_SCHEDULER_LOG)"; \
	fi
	@if [ -d "$(BENCHMARK_STATS)" ]; then \
		rm -rf "$(BENCHMARK_STATS)"; \
		echo "已清理测试统计: $(BENCHMARK_STATS)"; \
	else \
		echo "测试统计不存在: $(BENCHMARK_STATS)"; \
	fi
	@if [ -d "$(REQUEST_OUTPUT)" ]; then \
		rm -rf "$(REQUEST_OUTPUT)"; \
		echo "已清理请求输出: $(REQUEST_OUTPUT)"; \
	else \
		echo "请求输出不存在: $(REQUEST_OUTPUT)"; \
	fi