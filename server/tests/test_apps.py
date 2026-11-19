from hidagent.apps import app_hints


def test_hints_name_the_english_app_for_each_chinese_name_in_the_task():
    assert app_hints("打开备忘录") == "App names on this iPhone (search these, not the Chinese): 备忘录 = Notes"
    assert "哔哩哔哩 = bilibili" in app_hints("打开哔哩哔哩，搜索长安姑娘")
    hint = app_hints("打开设置，进入通用，再进入关于本机，然后回到主屏幕，打开备忘录")
    assert "设置 = Settings" in hint and "备忘录 = Notes" in hint


def test_longer_names_win_and_are_not_matched_twice():
    hint = app_hints("打开语音备忘录")
    assert "语音备忘录 = Voice Memos" in hint and "备忘录 = Notes" not in hint
    assert app_hints("打开谷歌翻译").count("=") == 1  # not also 翻译 = Translate


def test_no_app_name_no_hint():
    assert app_hints("128 乘以 37 等于多少") == ""
    assert app_hints("open Safari") == ""


def test_everyday_words_count_only_after_an_opening_verb():
    assert app_hints("打开设置，查看设备信息").endswith(": 设置 = Settings")
    assert "信息 = Messages" in app_hints("打开信息，看最新一条")
    assert "时钟 = Clock" in app_hints("打开时钟，切换到世界时钟页面，告诉我列出了哪些城市")
    assert app_hints("帮我查找一下") == ""


def test_third_party_apps_go_through_the_app_library():
    hint = app_hints("打开哔哩哔哩，搜索长安姑娘")
    assert "哔哩哔哩 = bilibili (third-party: open it from the App Library" in hint
    assert "App Library" not in app_hints("打开备忘录")
