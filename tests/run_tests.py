#!/usr/bin/env python3
"""
测试运行脚本
提供便捷的测试执行入口
"""
import subprocess
import sys
import os


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行所有测试")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode


def run_unit_tests():
    """运行单元测试"""
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode


def run_integration_tests():
    """运行集成测试"""
    print("=" * 60)
    print("运行集成测试")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode


def run_algorithm_tests():
    """运行算法测试"""
    print("=" * 60)
    print("运行算法测试")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/algorithms/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode


def run_e2e_tests():
    """运行端到端测试"""
    print("=" * 60)
    print("运行端到端测试")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/e2e/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode


def run_with_coverage():
    """运行测试并生成覆盖率报告"""
    print("=" * 60)
    print("运行测试并生成覆盖率报告")
    print("=" * 60)
    
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/",
            "--cov=app", "--cov=backend",
            "--cov-report=html", "--cov-report=term",
            "-v"
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode


def print_usage():
    """打印使用说明"""
    print("""
AudioMOS 测试运行脚本

用法: python run_tests.py [选项]

选项:
    all         运行所有测试
    unit        运行单元测试
    integration 运行集成测试
    algorithm   运行算法测试
    e2e         运行端到端测试
    coverage    运行测试并生成覆盖率报告
    help        显示此帮助信息

示例:
    python run_tests.py all
    python run_tests.py unit
    python run_tests.py coverage
""")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "all":
        return run_all_tests()
    elif command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "algorithm":
        return run_algorithm_tests()
    elif command == "e2e":
        return run_e2e_tests()
    elif command == "coverage":
        return run_with_coverage()
    elif command in ["help", "-h", "--help"]:
        print_usage()
        return 0
    else:
        print(f"未知命令: {command}")
        print_usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
