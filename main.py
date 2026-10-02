"""
Smart Market Analysis Bot - Big_Brother_ai_signal
Install:  pip install python-telegram-bot aiohttp numpy pandas
Run:      python signal_bot.py
"""
import asyncio
import datetime as dt
import sqlite3
from typing import Dict, List, Optional, Tuple

import aiohttp  
import numpy as np
import pandas as pd
from telegram import InlineKeyboardButton as B, InlineKeyboardMarkup as M, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

# ==================== CONFIG ====================
BOT_TOKEN = "8857147364:AAEmPxY0mzTcZdDzY-XC_HD5Vz23fqeYXec"
ADMIN_IDS = {8857147364}            # Your Telegram user id(s)
DB_PATH = "bot.db"
API_BASE = "https://quotexcandles.bdtraderpro.xyz/proversion/quotexcandles/Qx.php"
FREE_LIMIT, PREMIUM_LIMIT = 3, 30
MIN_SCORE = 3                      # Score threshold for signal
PAIRS = ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "AUDUSD_otc", "USDCAD_otc",
         "XAUUSD_otc", "BTCUSD_otc", "ETHUSD_otc", "USDBDT_otc", "USDINR_otc"]

busy: Dict[int, bool] = {}


# ==================== DATABASE ====================
def db():
    return sqlite3.connect(DB_PATH)


def setup_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY, name TEXT DEFAULT '',
            premium_until TEXT DEFAULT '', daily INTEGER DEFAULT 0, day TEXT DEFAULT '',
            wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0)""")


def get_user(uid: int, name: str = "") -> dict:
    today = dt.date.today().isoformat()
    with db() as c:
        c.execute("INSERT OR IGNORE INTO users(user_id,name,day) VALUES(?,?,?)", (uid, name, today))
        c.execute("UPDATE users SET daily=0, day=? WHERE user_id=? AND day!=?", (today, uid, today))
        r = c.execute("SELECT user_id,name,premium_until,daily,wins,losses FROM users WHERE user_id=?",
                      (uid,)).fetchone()
    return dict(zip(["id", "name", "premium_until", "daily", "wins", "losses"], r))


def is_premium(u: dict) -> bool:
    return u["id"] in ADMIN_IDS or (u["premium_until"] and u["premium_until"] >= dt.date.today().isoformat())


def limit_for(u: dict) -> int:
    return 10**6 if u["id"] in ADMIN_IDS else PREMIUM_LIMIT if is_premium(u) else FREE_LIMIT


def add_result(uid: int, win: bool):
    col = "wins" if win else "losses"
    with db() as c:
        c.execute(f"UPDATE users SET {col}={col}+1 WHERE user_id=?", (uid,))


def use_signal(uid: int):
    with db() as c:
        c.execute("UPDATE users SET daily=daily+1 WHERE user_id=?", (uid,))


# ==================== ANALYSIS ENGINE (SMC, MMC, Price Action & Indicators) ====================
def ema(x: np.ndarray, n: int) -> np.ndarray:
    a, out = 2 / (n + 1), np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(x: np.ndarray, n: int = 14) -> float:
    d = np.diff(x)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d)):
        au, ad = (au * (n - 1) + up[i]) / n, (ad * (n - 1) + dn[i]) / n
    return 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)


def analyze_advanced(df: pd.DataFrame) -> Tuple[int, List[str]]:
    score = 0
    reasons = []
    
    if len(df) < 30:
        return score, reasons

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    body = abs(c - o)
    
    # 1. Price Action: Engulfing & Pin Bars
    if prev["close"] < prev["open"] and c > o and c > prev["open"]:
        score += 2; reasons.append("Bullish Engulfing")
    elif prev["close"] > prev["open"] and c < o and c < prev["open"]:
        score -= 2; reasons.append("Bearish Engulfing")

    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    if lower_shadow > (2 * body) and upper_shadow < body:
        score += 2; reasons.append("Pin Bar Rejection (Bullish)")
    elif upper_shadow > (2 * body) and lower_shadow < body:
        score -= 2; reasons.append("Pin Bar Rejection (Bearish)")

    # 2. SMC & MMC Concepts (Liquidity Sweep & Order Block simulation via extremes)
    recent_high = df["high"].iloc[-25:-1].max()
    recent_low = df["low"].iloc[-25:-1].min()
    
    if l <= recent_low * 1.0001 and c > o:
        score += 2; reasons.append("SMC Support / Liquidity Grab")
    elif h >= recent_high * 0.9999 and c < o:
        score -= 2; reasons.append("SMC Resistance / Supply Zone")

    return score, reasons


def analyze(candles: List[dict]) -> Tuple[Optional[str], int, dict]:
    if len(candles) < 60:
        return None, 0, {}
    
    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    
    arr_c = df["close"].values
    e8, e21, e50 = ema(arr_c, 8)[-1], ema(arr_c, 21)[-1], ema(arr_c, 50)[-1]
    macd = ema(arr_c, 12) - ema(arr_c, 26)
    hist = (macd - ema(macd, 9))[-1]
    r = rsi(arr_c)
    mid, sd = arr_c[-20:].mean(), arr_c[-20:].std()
    price = arr_c[-1]

    score, why = 0, []
    
    # Indicator Votes
    if e8 > e21 > e50:
        score += 2; why.append("EMA Uptrend")
    elif e8 < e21 < e50:
        score -= 2; why.append("EMA Downtrend")
    if hist > 0:
        score += 1; why.append("MACD Bullish")
    elif hist < 0:
        score -= 1; why.append("MACD Bearish")
    if r < 30:
        score += 1; why.append(f"RSI Oversold ({r:.0f})")
    elif r > 70:
        score -= 1; why.append(f"RSI Overbought ({r:.0f})")
    if price <= mid - 2 * sd:
        score += 1; why.append("Lower Bollinger Band")
    elif price >= mid + 2 * sd:
        score -= 1; why.append("Upper Bollinger Band")

    # Combine SMC, MMC & Price Action
    adv_score, adv_reasons = analyze_advanced(df)
    score += adv_score
    why.extend(adv_reasons)

    direction = "CALL" if score >= MIN_SCORE else "PUT" if score <= -MIN_SCORE else None
    return direction, score, {"rsi": round(r, 1), "why": why}


# ==================== DATA FETCH ====================
async def fetch_candles(pair: str, count: int = 100) -> List[dict]:
    url = f"{API_BASE}?pair={pair}&timeframe=M1&count={count}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=12)) as r:
                data = (await r.json(content_type=None)).get("data", [])
        return sorted(data, key=lambda k: int(k.get("epoch", 0)))
    except Exception as e:
        print("fetch error:", e)
        return []


# ==================== TELEGRAM INTERFACE ====================
def home_kb():
    return M([[B("📊 New Signal", callback_data="pairs")],
              [B("👤 Profile", callback_data="profile")]])


async def run_signal(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, uid: int, pair: str):
    try:
        candles = await fetch_candles(pair)
        direction, score, info = analyze(candles)
        name = pair.replace("_otc", "")
        if not candles:
            await ctx.bot.send_message(chat_id, "❌ বাজার থেকে ডেটা পাওয়া যায়নি।", reply_markup=home_kb())
            return
        if direction is None:
            await ctx.bot.send_message(
                chat_id, f"⚪ {name}: NO TRADE\nमार्কেট পরিষ্কার নয় (Score: {score:+d})। অন্য পেয়ার চেক করুন।",
                reply_markup=home_kb())
            return

        entry = (dt.datetime.now() + dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
        use_signal(uid)
        icon = "🟢" if direction == "CALL" else "🔴"
        
        await ctx.bot.send_message(
            chat_id,
            f"👑 **Big_Brother_ai_signal**\n"
            f"{icon} {name} OTC — {direction}\n"
            f"⏰ Entry: {entry:%H:%M} | Expiry: M1\n"
            f"📊 Strength Score: {score:+d}\n"
            f"📈 RSI: {info['rsi']} | Reasons: {', '.join(info['why'])}\n\n⏳ ফলাফল যাচাই করা হচ্ছে...")

        await asyncio.sleep(max(0, (entry + dt.timedelta(minutes=1, seconds=8) - dt.datetime.now()).total_seconds()))

        res = await fetch_candles(pair, 10)
        target = next((k for k in res if abs(int(k["epoch"]) - int(entry.timestamp())) <= 30), None)
        if not target:
            await ctx.bot.send_message(chat_id, "⚠️ রেজাল্ট ক্যান্ডেল পাওয়া যায়নি।", reply_markup=home_kb())
            return
        o, cl = float(target["open"]), float(target["close"])
        win = cl > o if direction == "CALL" else cl < o
        add_result(uid, win)
        
        await ctx.bot.send_message(chat_id, f"{'✅ WIN' if win else '❌ LOSS'} — {name} {direction}\n"
                                            f"Open: {o} → Close: {cl}", reply_markup=home_kb())
    finally:
        busy.pop(uid, None)


async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    get_user(u.id, u.full_name)
    await update.message.reply_text(
        "👋 স্বাগতম **Smart Market Analysis Bot**-এ!\n"
        "প্রস্তুতকর্তა: **Big_Brother_ai_signal**\n\n"
        "📊 Technical Analysis • SMC • MMC • Price Action\n"
        "⚡ ডেমো অ্যাকাউন্টে টেস্ট করে মার্কেটে ট্রেড করুন।", reply_markup=home_kb(), parse_mode="Markdown")


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    u = q.from_user
    user = get_user(u.id, u.full_name)
    await q.answer()
    chat_id = q.message.chat_id

    if q.data == "pairs":
        rows = [[B(p.replace("_otc", ""), callback_data=f"p_{p}") for p in PAIRS[i:i + 2]]
                for i in range(0, len(PAIRS), 2)]
        await ctx.bot.send_message(chat_id, "💎 পেয়ার সিলেক্ট করুন:", reply_markup=M(rows))

    elif q.data.startswith("p_"):
        if busy.get(u.id):
            await q.answer("⏳ আগের সিগন্যালের ফলাফল আসা পর্যন্ত অপেক্ষা করুন", show_alert=True)
            return
        if user["daily"] >= limit_for(user):
            await ctx.bot.send_message(chat_id, "❌ আজকের দৈনিক সিগন্যাল লিমিট শেষ।")
            return
        busy[u.id] = True
        await ctx.bot.send_message(chat_id, "🔍 SMC, MMC এবং Price Action বিশ্লেষণ করা হচ্ছে...")
        asyncio.create_task(run_signal(ctx, chat_id, u.id, q.data[2:]))

    elif q.data == "profile":
        t = user["wins"] + user["losses"]
        wr = round(user["wins"] / t * 100, 1) if t else 0
        plan = "ADMIN" if u.id in ADMIN_IDS else "PREMIUM" if is_premium(user) else "FREE"
        lim = "∞" if u.id in ADMIN_IDS else limit_for(user)
        await ctx.bot.send_message(
            chat_id, f"👤 {user['name']}\n🆔 {u.id}\n👑 Brand: Big_Brother_ai_signal\n💎 Plan: {plan}\n"
                     f"📊 আজ ব্যবহৃত: {user['daily']}/{lim}\n✅ Wins: {user['wins']} | ❌ Losses: {user['losses']} | Win Rate: {wr}%",
            reply_markup=home_kb())


async def on_addpremium(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    try:
        uid, days = int(ctx.args[0]), int(ctx.args[1])
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /addpremium <user_id> <days>")
        return
    get_user(uid)
    until = (dt.date.today() + dt.timedelta(days=days)).isoformat()
    with db() as c:
        c.execute("UPDATE users SET premium_until=? WHERE user_id=?", (until, uid))
    await update.message.reply_text(f"✅ User {uid} granted premium until {until}")


async def on_users(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    with db() as c:
        n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    await update.message.reply_text(f"👥 Total Users: {n}")


def main():
    setup_db()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("addpremium", on_addpremium))
    app.add_handler(CommandHandler("users", on_users))
    app.add_handler(CallbackQueryHandler(on_callback))
    print("✅ Smart Market Analysis Bot (Big_Brother_ai_signal) started successfully!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
