from textwrap import TextWrapper

wrapper = TextWrapper(width=100, max_lines=5)

text = "".join(["Foobar..."] * 1000)

wrapped_lines = wrapper.wrap(text)
wrapper_text = """\n<mark>&gt;wrap&lt;</mark> """.join(wrapped_lines)
print(wrapper_text)

print(text)

print("done")
