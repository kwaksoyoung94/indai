# 상수 정의
DIVIDE_BY_ZERO_ERROR = "Cannot divide by 0"


# 사칙연산 함수 정의
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        return DIVIDE_BY_ZERO_ERROR  # 0으로 나눌 경우
    return a / b


def power(a, b):
    return a ** b


def mod(a, b):
    return a % b


# 테스트 코드 추출
def test_operations(num1, num2):
    operations = {
        "addition": add,
        "subtraction": subtract,
        "multiplication": multiply,
        "division": divide,
        "power": power,
        "modulo": mod,
    }

    for operation, func in operations.items():
        print(f"{operation.capitalize()} of {num1} and {num2}: {func(num1, num2)}")
    # 나누기 0 테스트
    print(f"Division of {num1} and 0: {divide(num1, 0)}")


if __name__ == "__main__":
    # 테스트 값
    num1 = 10
    num2 = 0

    # 결과 출력
    test_operations(num1, num2)