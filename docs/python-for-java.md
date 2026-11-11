# Python for Java 开发者

写这个项目时遇到的 Python 写法，按"Java 里的 X → Python 里的 Y"整理。代码示例尽量取自项目本身。

> 基于 Python 3.12（项目的 `.venv`）。

## 1. 基本数据结构

| Java | Python | 字面量 |
| --- | --- | --- |
| `ArrayList` | `list` | `[1, 2]` |
| `HashMap<String, Object>` | `dict` | `{"k": "v"}` |
| `HashSet` | `set` | `{1, 2}`（空集合必须写 `set()`，`{}` 是空 dict） |
| 不可变的小数组 / `record` 的轻量替代 | `tuple` | `(1, "a")` |
| `null` / `true` / `false` | `None` / `True` / `False` | |

`json.loads` 读出来的都是内置类型，不会映射成对象：JSON 数组 → `list`，对象 → `dict`。所以 `assignments.json` 读进来是 `list[dict]`。

取 dict 的值：

```python
a["course"]       # key 不存在抛 KeyError —— key 是自己定义的、一定存在时用这个，写错能马上发现
a.get("course")   # key 不存在返回 None —— 外部数据、字段可能缺失时用这个
```

## 2. Stream → 推导式

```java
list.stream().filter(a -> a.getCourse().equals(course)).collect(toList());
list.stream().map(a -> a.getName()).collect(toList());
list.stream().filter(...).findFirst().orElse(null);
list.stream().anyMatch(...);
list.stream().sorted(comparing(A::getDueAt)).collect(toList());
```

```python
[a for a in lst if a["course"] == course]          # filter
[a["name"] for a in lst]                            # map
next((a for a in lst if a["id"] == aid), None)      # findFirst().orElse(null)
any(a["course"] == course for a in lst)             # anyMatch
sorted(lst, key=lambda a: a["due_at"])              # sorted
sum(t["passed"] for t in results)                   # 计数：True 当 1，False 当 0
{c["id"]: c["name"] for c in courses}               # 转 Map：dict 推导式
```

读法：对 `lst` 里的每个 `a`，满足 `if` 就收集 `for` 前面的表达式。

## 3. 运算符重载

Java 不支持运算符重载（只有 `+` 对 String 是特例）。Python 里 `a / b` 实际调用 `a.__truediv__(b)`，每个类可以自己定义，所以**同一个运算符的含义由左边对象的类型决定**。

| 表达式 | 类型 | 含义 |
| --- | --- | --- |
| `7 / 2` | 数字 | 除法，结果 `3.5`（总是浮点数） |
| `7 // 2` | 数字 | 整除，结果 `3`（相当于 Java int 的 `/`） |
| `Path("a") / "b"` | `Path` | 拼接路径（相当于 `path.resolve("b")`） |
| `a - b` | `set` | 差集 |
| `a <= b` | `set` | a 是 b 的子集 |
| `a \| b`、`a & b` | `set` | 并集、交集 |
| `chunk1 + chunk2` | `AIMessageChunk` | 拼接流式输出的消息 |

## 4. 函数和参数

### 调用时的 `*` 和 `**`：拆开传参

```python
args = {"course": "软件工程"}
tools.list_assignments(**args)        # 等价于 list_assignments(course="软件工程")

nums = [1, 2]
pow(*nums)                            # 等价于 pow(1, 2)
```

`**` 是把字典的**键当参数名、值当参数值**，按关键字传。多出来的键、拼错的键都会报 `TypeError: got an unexpected keyword argument`。

### 定义时的 `*args` / `**kwargs`：收集参数（可变参数）

```python
def f(*args, **kwargs):
    print(args)     # 多余的位置参数 → tuple，相当于 Java 的 Object... args
    print(kwargs)   # 多余的关键字参数 → dict，Java 没有对应
```

| 位置 | `*` | `**` |
| --- | --- | --- |
| 调用时 `f(*x)` | 把列表拆开，按位置传 | 把字典拆开，按参数名传 |
| 定义时 `def f(*a)` | 收集成 tuple | 收集成 dict |

### 没有方法重载

`def` 本质上是给名字赋值，同名函数写两次，后一个覆盖前一个：

```python
def f(x: int): return "int"
def f(x: str): return "str"     # 覆盖了上面那个
f(1)                            # 'str'
```

要"根据参数类型做不同的事"：参数用联合类型 `A | B`，函数里用 `isinstance` 或检查字段分支。LangChain 的 `BaseTool.invoke` 就是这样，靠 `x.get("type") == "tool_call"` 区分输入（TS 里叫可辨识联合）。

- `@typing.overload`：只给类型检查器和 IDE 看，运行时仍是最后那个实现。
- `functools.singledispatch`：按第一个参数的**类型**分发，最接近 Java 重载。

### 返回多个值

```python
return msg.content, tokens          # 返回 tuple
answer, tokens = run(q)             # 调用方拆开
```

值一多容易拆错位置（项目里出过：两个 `return` 一个返回 2 个值、一个返回 3 个值）。超过两三个值时改用 dataclass（见下一节）。

## 5. 类和 dataclass

### dataclass ≈ record

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class RunResult:
    answer: str | None
    tokens: int = 0
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
```

- **`field(default_factory=list)`**：不能写 `= []`，否则所有实例共用同一个 list。这是 Python 最常见的坑之一（函数默认参数 `def f(x=[])` 也一样）。
- **`slots=True`**：只允许声明过的字段。不加的话，`result.step = 1`（少了个 s）不会报错，而是悄悄新建一个属性。

### 特殊方法（dunder methods）

| Java | Python |
| --- | --- |
| `toString()` | `__str__`（`print(x)` 调用） |
| 调试用的表示 | `__repr__`（REPL 里直接输入 `x` 显示的，一般带类名） |
| `equals()` | `__eq__` |
| 运算符 | `__truediv__`、`__add__`、`__sub__`、`__le__` ... |

pydantic 对象（如 `AIMessage`）的 `__str__` 输出 `字段=值 字段=值`，值是 Python 字典格式（单引号、`None`），不是 JSON。要 JSON 用 `.model_dump_json(indent=2)`。

## 6. 变量和作用域

- **变量只是名字，可以随时绑定到别的对象，类型也能变。** Java 同一作用域重复声明会编译报错，Python 没有任何提示。项目里出过：`result = RunResult(...)` 之后，又在循环里 `result = fn(...)`，后面 `result.tool_calls` 就变成了对 list 取属性，报 `AttributeError: 'list' object has no attribute 'tool_calls'`。
- **`global x` 只在给全局变量赋值时需要**，只读不用写。
- **docstring 必须是函数体的第一条语句**，前面放了别的语句（比如 `global`），它就只是一个没人用的字符串。
- 不要用内置函数名当变量名：`id`、`list`、`type`、`input` 等会被覆盖。

## 7. 类型标注

- 默认**只是提示**，运行时不检查（PyCharm 会检查并标出不一致）。
- 但有的框架会在运行时按它校验：pydantic、LangChain 的 `@tool`。`deadline: str` 时传 `None` 会直接抛 `ValidationError`，函数里处理 `None` 的代码执行不到。
- 写法：`str | None`（可空）、`list[dict]`、`tuple[str, int]`、`Annotated[str | None, "课程名"]`（附加元数据，`@tool` 会读成参数描述）。
- 建议都写上：写起来更像 Java，PyCharm 能提前抓错误。

## 8. 异常

```python
try:
    due = datetime.fromisoformat(deadline)      # try 只包住会抛异常的那一行
except ValueError as e:
    raise ValueError(f"deadline 不是 ISO 格式: {deadline}") from e
```

- 用具体的异常类型：参数值不对 `ValueError`，类型不对 `TypeError`（类比 `IllegalArgumentException`），不要直接 `raise Exception`。
- `raise ... from e`：保留原始异常，相当于 `new XxxException(msg, cause)`。
- `except` 只接你预料到的类型；要兜底（比如工具调用返回错误给模型）才用 `except Exception`。

## 9. 文件和路径

| Java | Python |
| --- | --- |
| try-with-resources | `with open(...) as f:` |
| `Files.readString(p)` | `p.read_text(encoding="utf-8")` |
| `Files.writeString(p, s)` | `p.write_text(s, encoding="utf-8")` |
| `path.resolve("x")` | `path / "x"` |
| `Files.createDirectories(p)` | `p.mkdir(parents=True, exist_ok=True)` |

```python
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "assignments.json"
DATA_FILE.write_text(json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8")
```

- **`encoding="utf-8"` 一定要写**，否则用系统默认编码，Windows 上中文可能乱码。
- **`ensure_ascii=False`**：否则中文被写成 `软件`。
- `json.dumps` 返回字符串，`json.dump` 直接写进文件对象。
- **相对路径是相对于运行时所在的目录**（cwd），不是代码文件所在的目录。项目里出过：`.env` 里 `ASSIGNMENT_DIR=./data`，换个目录运行就找不到数据，而且不报错，只是返回空列表。用 `Path(__file__)` 定位。
- `Path("~/x")` 不会自动展开 `~`，要 `.expanduser()`。

## 10. 日期时间

| Java | Python |
| --- | --- |
| `LocalDateTime.now()` | `datetime.now()`（不带时区） |
| `ZonedDateTime.now(ZoneOffset.UTC)` | `datetime.now(timezone.utc)` |
| `ZonedDateTime.parse(s)` | `datetime.fromisoformat(s)`（3.11+ 支持结尾的 `Z`） |
| `Duration` | `timedelta`（两个 datetime 相减得到） |
| `ZoneId.of("Asia/Shanghai")` | `ZoneInfo("Asia/Shanghai")` |

- **带时区和不带时区的 datetime 不能相减**：`TypeError: can't subtract offset-naive and offset-aware datetimes`。
- `"2026-11-22T15:59:59Z"` 和 `"2026-11-22T23:59:59+08:00"` 是同一时刻。解析后比较是相等的，但**字符串比较不相等**，排序也可能错。
- `timedelta` 为负时 `.days` 反直觉：过期 1 小时是 `days=-1, seconds=82800`。先用 `.total_seconds() < 0` 判断。
- 去掉微秒：`.replace(microsecond=0)` 或 `.isoformat(timespec="seconds")`。

## 11. 模块、入口和运行

- **`if __name__ == "__main__":`** 相当于 `main` 方法。直接运行文件时执行，被 `import` 时不执行。模块顶层的其他代码在 import 时都会执行。
- import 放在文件顶部。
- 用包路径导入：`from homework_agent import tools`。直接 `import tools` 只有在直接运行同目录的脚本时才碰巧能用。
- **`uv run xxx`**：在项目的 `.venv` 里执行，相当于 Maven/Gradle 帮你准备好 classpath。
- `[project.scripts]` 里的入口改了以后，`uv sync` 重新生成启动脚本。
- `sys.executable` 是当前 Python 解释器的路径，用 `subprocess` 运行另一个脚本时用它。

## 12. 装饰器 ≈ 注解，但更强

```python
@dataclass
@pytest.fixture
@tool
@mcp.tool()
```

Java 注解只是元数据，要靠框架反射读取。Python 装饰器是**一个函数**，接收被装饰的函数/类，返回一个**新东西替换它**：

- `@tool` 之后，名字指向的是 `StructuredTool` 对象，不再是函数，`list_assignments()` 会报 `'StructuredTool' object is not callable`。
- `@mcp.tool()` 带括号：先调用 `mcp.tool()` 得到装饰器，再装饰函数（工具要注册到 `mcp` 这个具体的 server 上）。

## 13. 测试：pytest ≈ JUnit

| JUnit | pytest |
| --- | --- |
| `@Test void testXxx()` | 普通函数 `def test_xxx():` |
| `assertEquals(a, b)` | `assert a == b`（失败时会打印两边的值） |
| `@BeforeEach` | `@pytest.fixture` |
| `@ParameterizedTest` + `@CsvSource` | `@pytest.mark.parametrize` |
| `assertThrows(...)` | `with pytest.raises(ValueError, match="..."):` |
| 注入固定的 `Clock` | `monkeypatch.setattr(tools, "datetime", FixedDatetime)` |

- **按命名约定自动发现**：`test_*.py` 文件里 `test_` 开头的函数。
- **fixture 按参数名注入**：测试函数参数叫 `data_dir`，pytest 就先运行同名 fixture 再把结果传进来（类似 Spring 按名字注入）。
- `tmp_path`、`monkeypatch` 是内置 fixture。`monkeypatch` 的替换在测试结束后自动还原。
- 常用：`-v` 详细、`-k name` 按名字筛选、`-x` 失败即停、`-s` 显示 print。

## 14. 看不懂一个对象时

- `type(x)`：是什么类型
- `dir(x)`：有哪些属性和方法
- `repr(x)`：带类名的表示
- PyCharm 里 Ctrl+左键跳进源码：很多"魔法"（如 `Path` 的 `/`）看一眼 `__truediv__` 就明白了
- 框架替你生成的东西打印出来确认（如 `tool.args`），不要假设
