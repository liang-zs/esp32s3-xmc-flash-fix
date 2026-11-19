"""Chinese app names -> the names this iPhone shows (system language English).

Tasks come in Chinese ("打开备忘录") but Spotlight and the Home Screen show English names, and searching the
Chinese word finds nothing useful. `app_hints` picks the names a task mentions so the planner and the actor are
told exactly what to search for. Third-party names are the ones on this iPhone 11's screens (2026-11-18).
"""

import re

APPS: dict[str, str] = {
    # Apple apps
    "设置": "Settings", "备忘录": "Notes", "照片": "Photos", "相册": "Photos", "提醒事项": "Reminders",
    "日历": "Calendar", "时钟": "Clock", "地图": "Maps", "文件": "Files", "通讯录": "Contacts",
    "联系人": "Contacts", "信息": "Messages", "短信": "Messages", "电话": "Phone", "相机": "Camera",
    "计算器": "Calculator", "健康": "Health", "钱包": "Wallet", "天气": "Weather", "邮件": "Mail",
    "音乐": "Music", "快捷指令": "Shortcuts", "语音备忘录": "Voice Memos", "查找": "Find My",
    "Safari浏览器": "Safari", "浏览器": "Safari", "股市": "Stocks", "图书": "Books", "播客": "Podcasts",
    "视频": "TV", "家庭": "Home", "无边记": "Freeform", "可立拍": "Clips", "翻译": "Translate",
    "指南针": "Compass", "测距仪": "Measure", "放大器": "Magnifier", "提示": "Tips", "健身": "Fitness",
    "应用商店": "App Store", "App Store": "App Store",
}
# third-party apps on this phone. Spotlight is unreliable for them: for bilibili its Top Hit was the app's page in
# Settings (gear icon) and the rest web searches (iPhone 11, 2026-11-19); the App Library search lists apps only.
THIRD_PARTY: dict[str, str] = {
    "哔哩哔哩": "bilibili", "B站": "bilibili", "b站": "bilibili", "小红书": "rednote", "微信": "WeChat",
    "支付宝": "Alipay", "携程": "Trip.com", "谷歌翻译": "Google Translate", "谷歌地图": "Google Maps",
    "亚马逊": "Amazon", "脸书": "Facebook", "油管": "YouTube",
}
APPS.update(THIRD_PARTY)


# also everyday words ("查看设备信息" is no request for Messages): only an app name right after an opening verb
COMMON_WORDS = {"信息", "文件", "查找", "翻译", "提示", "视频", "家庭", "音乐", "电话", "地图", "时钟", "天气", "健康",
                "相机", "钱包", "健身", "照片", "图书", "浏览器", "短信", "股市"}
OPEN_VERB = r"(?:打开|启动|进入|去|用|在|点开|切换到)\s*"


def app_hints(task: str) -> str:
    """'备忘录 = Notes' for each app name in `task`, longest names first, so 语音备忘录 is not also 备忘录."""
    found, rest = [], task
    for zh in sorted(APPS, key=len, reverse=True):
        if zh == APPS[zh] or zh not in rest:
            continue
        if zh in COMMON_WORDS and not re.search(OPEN_VERB + re.escape(zh), rest):
            continue
        found.append(f"{zh} = {APPS[zh]}" + (" (third-party: open it from the App Library, not Spotlight)"
                                              if zh in THIRD_PARTY else ""))
        rest = rest.replace(zh, " ")
    return ("App names on this iPhone (search these, not the Chinese): " + "; ".join(found)) if found else ""
