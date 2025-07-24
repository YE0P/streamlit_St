import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import logging
import sys
import io

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ])

# --- 퀀트 분석 핵심 함수들 (이전과 동일) ---
# 데이터 수집 함수 (get_real_time_data)
@st.cache_data(ttl=3600) # 1시간 동안 데이터 캐싱
def get_real_time_data(ticker, current_date, specific_expiration=None, period="1y"):
    logging.info(f"데이터 로딩 시작: {ticker}, 기간: {period}")
    stock = yf.Ticker(ticker)

    current_price = None
    historical_data = None
    exp_date_obj = None
    option_df = pd.DataFrame()
    news_data = []

    try:
        hist = stock.history(period=period, interval="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            historical_data = hist.copy()
            logging.info(f"현재 주가 가져오기 성공: ${current_price:.2f} for {ticker}")
        else:
            logging.warning(f"과거 데이터를 가져오지 못했습니다: {ticker}")
            return None, None, pd.DataFrame(), None, []
    except Exception as e:
        logging.error(f"주가 및 과거 데이터를 가져오는 데 실패했습니다 for {ticker}: {e}")
        return None, None, pd.DataFrame(), None, []

    expirations = stock.options
    if not expirations:
        logging.warning(f"{ticker}에 대한 옵션 데이터를 찾을 수 없습니다.")
    else:
        selected_expiration_str = None
        if specific_expiration and specific_expiration in expirations:
            selected_expiration_str = specific_expiration
            logging.info(f"사용자가 지정한 만기일 '{specific_expiration}'을 선택했습니다 for {ticker}.")
        else:
            for exp_str in expirations:
                exp_date_obj_temp = datetime.strptime(exp_str, '%Y-%m-%d')
                if exp_date_obj_temp >= current_date.replace(hour=0, minute=0, second=0, microsecond=0):
                    selected_expiration_str = exp_str
                    logging.info(f"가장 가까운 만기일 '{selected_expiration_str}'을 선택했습니다 for {ticker}.")
                    break
                
        if selected_expiration_str is None:
            logging.error(f"{current_date.strftime('%Y-%m-%d')} 이후의 유효한 옵션 만기일을 찾을 수 없습니다 for {ticker}.")
        else:
            exp_date_obj = datetime.strptime(selected_expiration_str, '%Y-%m-%d')

            try:
                option_chain = stock.option_chain(selected_expiration_str)
                logging.info(f"옵션 체인 데이터 가져오기 성공 for {ticker}: 만기일 {selected_expiration_str}")

                calls_df = option_chain.calls[['strike', 'openInterest', 'impliedVolatility']]
                calls_df.columns = ['strike', 'call_oi', 'implied_volatility_call']
                puts_df = option_chain.puts[['strike', 'openInterest', 'impliedVolatility']]
                puts_df.columns = ['strike', 'put_oi', 'implied_volatility_put']

                full_option_df = pd.merge(calls_df, puts_df, on='strike', how='outer')
                full_option_df['call_oi'] = full_option_df['call_oi'].fillna(0)
                full_option_df['put_oi'] = full_option_df['put_oi'].fillna(0)
                full_option_df['implied_volatility'] = full_option_df[['implied_volatility_call', 'implied_volatility_put']].mean(axis=1).fillna(0)
                
                option_df = full_option_df[['strike', 'call_oi', 'put_oi', 'implied_volatility']].fillna(0)
                
                initial_rows = len(option_df)
                option_df = option_df[option_df['implied_volatility'] > 0.001].copy()
                if len(option_df) < initial_rows:
                    logging.info(f"IV가 0에 가까운 옵션 데이터 {initial_rows - len(option_df)}개 제거됨 for {ticker}.")

                if option_df.empty:
                    logging.warning(f"필터링 후 유효한 옵션 데이터가 없습니다 for {ticker}.")

            except Exception as e:
                logging.error(f"옵션 체인 데이터를 가져오는 데 실패했습니다 for {ticker} (만기일: {selected_expiration_str}): {e}")

    try:
        news_data = stock.news
        logging.info(f"뉴스 데이터 {len(news_data)}개 가져오기 성공 for {ticker}.")
    except Exception as e:
        logging.error(f"뉴스 데이터를 가져오는 데 실패했습니다 for {ticker}: {e}")
        news_data = []

    return current_price, exp_date_obj, option_df, historical_data, news_data

# 기술적 지표 계산 함수 (calculate_technical_indicators)
def calculate_technical_indicators(df):
    if df.empty:
        logging.warning("과거 주가 데이터가 없어 기술적 지표를 계산할 수 없습니다.")
        return df

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    return df

# 뉴스 분석 함수 (analyze_news_sentiment)
def analyze_news_sentiment(news_list, keywords_positive, keywords_negative):
    logging.info("뉴스 감성 분석 시작 (키워드 매칭).")
    sentiment = {"positive": 0, "negative": 0, "neutral": 0}
    
    if not news_list:
        logging.info("분석할 뉴스가 없습니다.")
        return sentiment

    for news_item in news_list:
        title = news_item.get('title', '').lower()
        text_to_analyze = title

        is_positive = any(keyword in text_to_analyze for keyword in keywords_positive)
        is_negative = any(keyword in text_to_analyze for keyword in keywords_negative)
        
        if is_positive and not is_negative:
            sentiment["positive"] += 1
        elif is_negative and not is_positive:
            sentiment["negative"] += 1
        else:
            sentiment["neutral"] += 1

    logging.info(f"뉴스 감성 분석 완료: 긍정 {sentiment['positive']}, 부정 {sentiment['negative']}, 중립 {sentiment['neutral']}")
    return sentiment

# 맥스페인(Max Pain) 계산 함수
def calculate_max_pain(option_chain_df, current_price_range):
    if option_chain_df.empty:
        return None, "옵션 체인 데이터가 비어 있습니다."

    max_pain_data = {}
    for price in current_price_range:
        total_option_value_at_expiration = 0
        for _, row in option_chain_df.iterrows():
            strike = row['strike']
            call_oi = row['call_oi']
            put_oi = row['put_oi']
            if price > strike: total_option_value_at_expiration += (price - strike) * call_oi
            if price < strike: total_option_value_at_expiration += (strike - price) * put_oi
        max_pain_data[price] = total_option_value_at_expiration

    if not max_pain_data:
        logging.error("맥스페인 계산을 위한 유효한 데이터 포인트를 찾을 수 없습니다.")
        return None, "해당 범위에서 맥스페인 계산을 위한 데이터가 없습니다."

    min_loss_strike = min(max_pain_data, key=max_pain_data.get)
    return min_loss_strike, max_pain_data[min_loss_strike]

# 예상무브(Expected Move) 계산 함수
def calculate_expected_move(option_chain_df, current_price, expiration_date, current_date):
    if option_chain_df.empty:
        return None, "옵션 체인 데이터가 비어 있습니다."

    atm_options = option_chain_df.iloc[(option_chain_df['strike'] - current_price).abs().argsort()[:2]]
    if atm_options.empty or atm_options['implied_volatility'].mean() == 0:
        return None, "ATM 옵션 또는 유효한 IV 없음."
        
    atm_iv = atm_options['implied_volatility'].mean()
    
    time_to_expiration_seconds = (expiration_date - current_date).total_seconds()
    if time_to_expiration_seconds <= 0:
        time_to_expiration_seconds = 3600 # 최소 1시간 (초)

    time_to_expiration_years = time_to_expiration_seconds / (365.25 * 24 * 60 * 60)
    expected_move = current_price * atm_iv * np.sqrt(time_to_expiration_years)
    
    return expected_move, atm_iv

# 매수/매도 신호 생성 함수
def generate_trade_signal(current_price, max_pain_price, expected_move_lower, expected_move_upper,
                             latest_rsi, sma_20, sma_50, sma_200, news_sentiment):
    MP_THRESHOLD_PERCENT = 0.015 
    EM_BUFFER_PERCENT = 0.005    
    RSI_OVERBOUGHT = 70          
    RSI_OVERSOLD = 30            

    buy_strength = 0
    sell_strength = 0
    signal = 0 

    # --- 매수 신호 조건 ---
    if max_pain_price is not None and current_price < max_pain_price * (1 - MP_THRESHOLD_PERCENT):
        buy_strength += 1
    if expected_move_lower is not None and current_price <= expected_move_lower * (1 + EM_BUFFER_PERCENT):
        buy_strength += 1
    if latest_rsi is not None and latest_rsi <= RSI_OVERSOLD:
        buy_strength += 1
    if sma_20 is not None and sma_50 is not None and current_price > sma_20 and sma_20 > sma_50:
        buy_strength += 1
    elif sma_20 is not None and current_price > sma_20:
         buy_strength += 0.5

    if news_sentiment is not None and news_sentiment["positive"] > news_sentiment["negative"]:
       buy_strength += 1

    # --- 매도 신호 조건 ---
    if max_pain_price is not None and current_price > max_pain_price * (1 + MP_THRESHOLD_PERCENT):
        sell_strength += 1
    if expected_move_upper is not None and current_price >= expected_move_upper * (1 - EM_BUFFER_PERCENT):
        sell_strength += 1
    if latest_rsi is not None and latest_rsi >= RSI_OVERBOUGHT:
        sell_strength += 1
    if sma_20 is not None and sma_50 is not None and current_price < sma_20 and sma_20 < sma_50:
        sell_strength += 1
    elif sma_20 is not None and current_price < sma_20:
        sell_strength += 0.5
    
    if news_sentiment is not None and news_sentiment["negative"] > news_sentiment["positive"]:
       sell_strength += 1

    if buy_strength >= 3 and sell_strength < 1:
        signal = 1
    elif sell_strength >= 3 and buy_strength < 1:
        signal = -1

    return signal, buy_strength, sell_strength # 신호 강도도 함께 반환

# 백테스팅 시스템 함수 (run_backtest)
def run_backtest(ticker, historical_data, initial_capital=100000, max_pain_price_func=None, expected_move_func=None, option_data_for_backtest=None):
    logging.info(f"백테스팅 시작 for {ticker}...")
    
    if historical_data.empty or len(historical_data) < 200:
        logging.error(f"백테스팅에 필요한 충분한 과거 주가 데이터가 없습니다 for {ticker} (최소 200일 필요).")
        return {}

    df = historical_data.copy()
    df = calculate_technical_indicators(df)
    df = df.dropna()

    if df.empty:
        logging.error(f"기술적 지표 계산 후 유효한 데이터가 남아있지 않습니다 for {ticker}. 백테스팅 불가.")
        return {}

    capital = initial_capital
    shares_held = 0
    trade_log = []
    
    current_option_df = option_data_for_backtest 
    temp_exp_date_offset_days = 7 

    for i in range(len(df)):
        current_date_for_day = df.index[i].to_pydatetime()
        current_price = df['Close'].iloc[i]
        
        latest_rsi = df['RSI'].iloc[i] 
        sma_20 = df['SMA_20'].iloc[i]
        sma_50 = df['SMA_50'].iloc[i]
        sma_200 = df['SMA_200'].iloc[i]

        max_pain_price = None
        expected_move_lower = None
        expected_move_upper = None

        if max_pain_price_func and current_option_df is not None and not current_option_df.empty:
            price_range_for_mp = np.arange(current_price * 0.85, current_price * 1.15, 0.5)
            mp_price, _ = max_pain_price_func(current_option_df, price_range_for_mp)
            max_pain_price = mp_price

        if expected_move_func and current_option_df is not None and not current_option_df.empty:
            temp_exp_date = current_date_for_day + timedelta(days=temp_exp_date_offset_days)
            em, _ = expected_move_func(current_option_df, current_price, temp_exp_date, current_date_for_day)
            if em is not None:
                expected_move_lower = current_price - em
                expected_move_upper = current_price + em
        
        # generate_trade_signal은 이제 신호 강도도 반환합니다.
        signal, _, _ = generate_trade_signal(current_price, max_pain_price, expected_move_lower, expected_move_upper,
                                         latest_rsi, sma_20, sma_50, sma_200, news_sentiment=None) # 백테스팅에서 뉴스 제외

        if signal == 1 and shares_held == 0:
            shares_to_buy = int(capital * 0.95 // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                capital -= cost
                shares_held += shares_to_buy
                trade_log.append({'Date': current_date_for_day, 'Type': 'BUY', 'Price': current_price, 'Shares': shares_to_buy, 'Capital': capital, 'Portfolio_Value': capital + shares_held * current_price})

        elif signal == -1 and shares_held > 0:
            revenue = shares_held * current_price
            capital += revenue
            trade_log.append({'Date': current_date_for_day, 'Type': 'SELL', 'Price': current_price, 'Shares': shares_held, 'Capital': capital, 'Portfolio_Value': capital})
            shares_held = 0

        if not trade_log or trade_log[-1]['Date'].date() != current_date_for_day.date():
             current_portfolio_value = capital + shares_held * current_price
             trade_log.append({'Date': current_date_for_day, 'Type': 'HOLD', 'Price': current_price, 'Shares': shares_held, 'Capital': capital, 'Portfolio_Value': current_portfolio_value})
        elif trade_log and trade_log[-1]['Date'].date() == current_date_for_day.date():
            trade_log[-1]['Portfolio_Value'] = capital + shares_held * current_price

    final_portfolio_value = capital + shares_held * df['Close'].iloc[-1]

    total_return = (final_portfolio_value - initial_capital) / initial_capital * 100

    start_date = df.index.min().to_pydatetime()
    end_date = df.index.max().to_pydatetime()
    num_years = (end_date - start_date).days / 365.25
    cagr = ((final_portfolio_value / initial_capital) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0

    buy_trades_count = 0
    sell_trades_count = 0
    winning_trades = 0
    
    open_trade = False
    buy_price = 0
    for trade in trade_log:
        if trade['Type'] == 'BUY':
            if not open_trade:
                buy_trades_count += 1
                buy_price = trade['Price']
                open_trade = True
        elif trade['Type'] == 'SELL':
            if open_trade:
                sell_trades_count += 1
                if trade['Price'] > buy_price:
                    winning_trades += 1
                open_trade = False
    
    total_completed_trades = min(buy_trades_count, sell_trades_count)
    win_rate = (winning_trades / total_completed_trades) * 100 if total_completed_trades > 0 else 0

    portfolio_values = pd.Series([t['Portfolio_Value'] for t in trade_log if t['Type'] != 'BUY'], index=[t['Date'] for t in trade_log if t['Type'] != 'BUY'])
    if not portfolio_values.empty:
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min() * 100
    else:
        max_drawdown = 0

    results = {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_portfolio_value": final_portfolio_value,
        "total_return_percent": total_return,
        "cagr_percent": cagr,
        "win_rate_percent": win_rate,
        "max_drawdown_percent": max_drawdown,
        "total_trades": total_completed_trades,
        "trade_log": trade_log
    }
    
    logging.info(f"백테스팅 완료 for {ticker}.")
    return results

# 시각화 함수 (plot_analysis_results)
@st.cache_data(ttl=3600)
def plot_analysis_results(ticker, current_price, max_pain_price, expected_move_lower, expected_move_upper, option_chain_df, historical_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    if not option_chain_df.empty:
        call_oi_df = option_chain_df[['strike', 'call_oi']]
        put_oi_df = option_chain_df[['strike', 'put_oi']]

        width = 0.4
        ax1.bar(call_oi_df['strike'] - width/2, call_oi_df['call_oi'], width, label='콜 미결제약정 (OI)', color='lightcoral', alpha=0.7)
        ax1.bar(put_oi_df['strike'] + width/2, put_oi_df['put_oi'], width, label='풋 미결제약정 (OI)', color='lightskyblue', alpha=0.7)
        ax1.set_xlabel('행사가 (Strike Price)', fontsize=10)
        ax1.set_ylabel('미결제약정 (Open Interest)', color='black', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc='upper left', fontsize=9)
    else:
        ax1.text(0.5, 0.5, '옵션 OI 데이터를 가져올 수 없습니다.', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

    ax1.axvline(current_price, color='green', linestyle='--', linewidth=2, label=f'현재 주가: ${current_price:.2f}')
    if max_pain_price:
        ax1.axvline(max_pain_price, color='purple', linestyle=':', linewidth=2, label=f'맥스페인: ${max_pain_price:.2f}')
    
    if expected_move_lower is not None and expected_move_upper is not None:
        ax1.axvspan(expected_move_lower, expected_move_upper, color='gray', alpha=0.2, label=f'예상무브 범위 ($1\sigma$): ${expected_move_lower:.2f} ~ ${expected_move_upper:.2f}')
    
    ax1.set_title(f'{ticker} 옵션 OI 및 주요 가격 분석', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=9)

    if 'RSI' in historical_data.columns and not historical_data['RSI'].isnull().all():
        ax2.plot(historical_data.index, historical_data['RSI'], color='orange', label='RSI (14일)')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='과매수 (70)')
        ax2.axhline(30, color='blue', linestyle='--', alpha=0.7, label='과매도 (30)')
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_xlabel('날짜', fontsize=10)
        ax2.set_title('RSI (Relative Strength Index)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'RSI 데이터를 계산할 수 없습니다.', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.suptitle(f'{ticker} 통합 분석', fontsize=16)
    
    return fig

# --- Gemini 분석 리포트 및 추천 종목 생성 함수 ---
def generate_gemini_report(ticker, current_price, max_pain_price, expected_move_lower, expected_move_upper,
                           latest_rsi, sma_20, sma_50, sma_200, news_sentiment, backtest_results, trade_signal_strength):
    
    report_content = []
    recommendation = "관망" # 기본 추천

    # 1. 시장 및 종목 개요
    report_content.append(f"## 📈 {ticker} 주식 Gemini 분석 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    report_content.append(f"### 현재 시장 상황 요약")
    report_content.append(f"**현재 {ticker} 주가:** ${current_price:.2f}")
    
    # 2. 옵션 시장 분석
    report_content.append("\n### 옵션 시장 분석 (Max Pain & Expected Move)")
    if max_pain_price is not None:
        report_content.append(f"- **맥스페인 가격:** ${max_pain_price:.2f}")
        if current_price > max_pain_price:
            report_content.append(f"  현재 주가는 맥스페인 가격보다 높습니다. 만기일이 임박할수록 주가가 맥스페인으로 **하향 수렴하려는 압력**이 있을 수 있습니다.")
        elif current_price < max_pain_price:
            report_content.append(f"  현재 주가는 맥스페인 가격보다 낮습니다. 만기일이 임박할수록 주가가 맥스페인으로 **상향 수렴하려는 압력**이 있을 수 있습니다.")
        else:
            report_content.append(f"  현재 주가는 맥스페인 가격에 근접합니다. 만기까지 해당 가격대에서 **횡보할 가능성**이 높습니다.")
    else:
        report_content.append("- 맥스페인 데이터를 분석할 수 없습니다 (옵션 데이터 부족).")

    if expected_move_lower is not None and expected_move_upper is not None:
        report_content.append(f"- **예상무브 범위 ($1\\sigma$):** ${expected_move_lower:.2f} ~ ${expected_move_upper:.2f}")
        report_content.append(f"  시장은 만기까지 주가가 이 범위 내에 있을 확률을 약 68.2%로 예상합니다. 이 범위를 벗어나는 움직임은 비정상적인 변동성으로 해석될 수 있습니다.")
    else:
        report_content.append("- 예상무브 데이터를 분석할 수 없습니다 (옵션 데이터 부족).")

    # 3. 기술적 지표 분석
    report_content.append("\n### 기술적 지표 분석")
    if latest_rsi is not None:
        report_content.append(f"- **RSI (14일):** {latest_rsi:.2f}")
        if latest_rsi >= 70:
            report_content.append("  RSI가 과매수 구간(70 이상)에 진입했습니다. 단기적인 가격 하락 조정 가능성을 시사합니다.")
        elif latest_rsi <= 30:
            report_content.append("  RSI가 과매도 구간(30 이하)에 진입했습니다. 단기적인 가격 반등 가능성을 시사합니다.")
        else:
            report_content.append("  RSI는 중립 구간에 있습니다. 뚜렷한 과매수/과매도 신호는 없습니다.")
    else:
        report_content.append("- RSI 데이터를 계산할 수 없습니다.")

    if sma_20 is not None and sma_50 is not None and sma_200 is not None:
        report_content.append(f"- **이동평균선:** SMA20(${sma_20:.2f}), SMA50(${sma_50:.2f}), SMA200(${sma_200:.2f})")
        if current_price > sma_20 and sma_20 > sma_50 and sma_50 > sma_200:
            report_content.append("  모든 이동평균선 위에 주가가 위치하며, 단기-중기-장기 이평선이 정배열입니다. 이는 **강력한 상승 추세**를 나타냅니다.")
        elif current_price < sma_20 and sma_20 < sma_50 and sma_50 < sma_200:
            report_content.append("  모든 이동평균선 아래에 주가가 위치하며, 역배열입니다. 이는 **강력한 하락 추세**를 나타냅니다.")
        elif current_price > sma_20 and sma_20 < sma_50:
            report_content.append("  주가가 단기 이평선 위에 있으나, 단기 이평선이 중기 이평선 아래에 있어 **추세 전환 가능성**을 시사합니다.")
        else:
            report_content.append("  이동평균선들이 혼조세를 보입니다. **추세가 불분명하거나 횡보장**일 수 있습니다.")
    else:
        report_content.append("- 이동평균선 데이터를 계산할 수 없습니다.")

    # 4. 뉴스 감성 분석
    report_content.append("\n### 뉴스 감성 분석")
    if news_sentiment["positive"] > 0 or news_sentiment["negative"] > 0:
        report_content.append(f"- 최근 뉴스에서 긍정 키워드 {news_sentiment['positive']}건, 부정 키워드 {news_sentiment['negative']}건이 감지되었습니다.")
        if news_sentiment["positive"] > news_sentiment["negative"] * 2:
            report_content.append("  긍정적인 뉴스가 압도적으로 많아 **투자 심리가 매우 긍정적**입니다.")
        elif news_sentiment["negative"] > news_sentiment["positive"] * 2:
            report_content.append("  부정적인 뉴스가 압도적으로 많아 **투자 심리가 매우 부정적**입니다.")
        elif news_sentiment["positive"] > news_sentiment["negative"]:
            report_content.append("  전반적으로 긍정적인 뉴스가 우세하여 **투자 심리가 다소 긍정적**입니다.")
        elif news_sentiment["negative"] > news_sentiment["positive"]:
            report_content.append("  전반적으로 부정적인 뉴스가 우세하여 **투자 심리가 다소 부정적**입니다.")
        else:
            report_content.append("  긍정/부정 뉴스가 혼재되어 **투자 심리가 혼조세**를 보입니다.")
    else:
        report_content.append("- 최신 뉴스 데이터를 가져오거나 분석할 수 없습니다.")

    # 5. 백테스팅 결과 요약
    report_content.append("\n### 백테스팅 결과 요약")
    if backtest_results and backtest_results['total_trades'] > 0:
        report_content.append(f"- **총 수익률:** {backtest_results['total_return_percent']:.2f}%")
        report_content.append(f"- **연평균 수익률 (CAGR):** {backtest_results['cagr_percent']:.2f}%")
        report_content.append(f"- **승률:** {backtest_results['win_rate_percent']:.2f}%")
        report_content.append(f"- **최대 낙폭 (MDD):** {backtest_results['max_drawdown_percent']:.2f}%")
        report_content.append(f"  과거 백테스팅 결과는 전략의 잠재력을 보여주지만, 이는 **미래 수익을 보장하지 않습니다.** 특히 옵션 데이터의 한계로 인해 맥스페인/예상무브 백테스팅은 참고용으로만 활용해야 합니다.")
    else:
        report_content.append("- 백테스팅 결과를 가져오거나 실행할 수 없습니다.")
        
    # 6. Gemini의 종합적인 투자 의견 및 추천 종목 (현재 분석 종목 위주)
    report_content.append("\n## ✨ Gemini의 종합 투자 의견 및 추천 종목")
    
    if trade_signal_strength == 1: # 강한 매수 신호 (buy_strength >= 3)
        report_content.append(f"**{ticker}에 대한 현재 종합 분석 결과, **강력한 매수 신호**가 감지되었습니다.**")
        report_content.append("다수의 지표(옵션 시장의 수렴 압력, 과매도 상태, 긍정적 뉴스 흐름 등)가 매수를 지지하고 있습니다.")
        report_content.append("단기적인 반등 또는 상승 추세 진입을 노려볼 수 있는 유리한 시점으로 판단됩니다. 손절매 라인을 엄격히 지키며 진입하는 것을 추천합니다.")
        recommendation = "강력 매수"
    elif trade_signal_strength == -1: # 강한 매도 신호 (sell_strength >= 3)
        report_content.append(f"**{ticker}에 대한 현재 종합 분석 결과, **강력한 매도 신호**가 감지되었습니다.**")
        report_content.append("다수의 지표(옵션 시장의 하락 압력, 과매수 상태, 부정적 뉴스 흐름 등)가 매도를 지지하고 있습니다.")
        report_content.append("현재 포지션을 정리하거나 공매도를 고려해 볼 수 있는 시점으로 판단됩니다. 상승 리스크를 관리하며 접근하는 것을 추천합니다.")
        recommendation = "강력 매도"
    else: # 뚜렷한 신호 없음 (buy_strength < 3 and sell_strength < 3)
        report_content.append(f"**{ticker}에 대한 현재 종합 분석 결과, **뚜렷한 매매 신호는 감지되지 않았습니다.**")
        report_content.append("여러 지표들이 혼조세를 보이거나 명확한 방향을 제시하지 않고 있습니다. 불확실성이 높으므로, **관망 포지션을 유지**하며 추가적인 시장 변화를 기다리는 것이 현명할 수 있습니다.")
        recommendation = "관망"
        
    report_content.append(f"\n**Gemini의 {ticker}에 대한 추천 의견: ** {recommendation}")
    report_content.append("\n*면책 조항: 이 리포트는 인공지능 분석에 기반하며, 실제 투자 결정은 전문가의 조언과 개인의 판단 하에 이루어져야 합니다. 모든 투자에는 원금 손실의 위험이 따릅니다.*")

    return "\n".join(report_content)


# --- Streamlit 앱의 UI 구성 ---

st.set_page_config(layout="wide", page_title="퀀트 주식 분석기")

st.title("📈 퀀트 주식 분석기")
st.markdown("---")

# 사이드바 입력 위젯
st.sidebar.header("분석 설정")
ticker_symbol = st.sidebar.text_input("주식 티커 (예: NVDA, AAPL, MSFT)", value="NVDA").upper()
backtest_period = st.sidebar.selectbox("백테스팅 기간", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
specific_expiration = st.sidebar.text_input("옵션 만기일 지정 (YYYY-MM-DD, 비워두면 가장 가까운 만기일)", value="")

# "분석 실행" 버튼
analyze_button = st.sidebar.button("분석 실행")

# --- 메인 콘텐츠 영역 ---
if analyze_button:
    if not ticker_symbol:
        st.error("분석할 주식 티커를 입력해주세요.")
    else:
        # 2025년 7월 25일 오전 1시 21분 08초 KST (현재 시간 설정)
        CURRENT_ANALYSIS_DATE = datetime.now() 
        
        st.header(f"{ticker_symbol} 통합 분석 및 전략 제안")
        st.info(f"분석 기준 시간: {CURRENT_ANALYSIS_DATE.strftime('%Y년 %m월 %d일 %H시 %M분')}")
        st.markdown("---")

        with st.spinner("데이터를 가져오고 분석 중입니다... 잠시만 기다려주세요."):
            current_price, expiration_date, option_df, historical_data, news_list = \
                get_real_time_data(ticker_symbol, CURRENT_ANALYSIS_DATE, specific_expiration, period=backtest_period)

            if current_price is None or historical_data.empty:
                st.error(f"**오류:** {ticker_symbol} 데이터 로딩 실패 또는 유효한 과거 주가 데이터 없음.")
                st.warning("yfinance는 과거 옵션 데이터를 제공하지 않으므로, 맥스페인/예상무브 백테스팅은 제한적입니다.")
            else:
                # 탭으로 결과 분리
                tab1, tab2, tab3, tab4 = st.tabs(["실시간 분석", "백테스팅 결과", "분석 차트", "Gemini 리포트 및 추천"])

                # 기술적 지표 계산 (모든 탭에서 사용될 수 있으므로 먼저 계산)
                historical_data_for_display = calculate_technical_indicators(historical_data.copy())
                latest_rsi = historical_data_for_display['RSI'].iloc[-1] if 'RSI' in historical_data_for_display.columns and not historical_data_for_display['RSI'].isnull().iloc[-1] else None
                latest_sma_20 = historical_data_for_display['SMA_20'].iloc[-1] if 'SMA_20' in historical_data_for_display.columns and not historical_data_for_display['SMA_20'].isnull().iloc[-1] else None
                latest_sma_50 = historical_data_for_display['SMA_50'].iloc[-1] if 'SMA_50' in historical_data_for_display.columns and not historical_data_for_display['SMA_50'].isnull().iloc[-1] else None
                latest_sma_200 = historical_data_for_display['SMA_200'].iloc[-1] if 'SMA_200' in historical_data_for_display.columns and not historical_data_for_display['SMA_200'].isnull().iloc[-1] else None

                # 뉴스 감성 분석
                positive_keywords = ['성장', '이익', '호재', '상승', '돌파', '신기술', '확장', '수주', '긍정적', '강력한', '성공']
                negative_keywords = ['하락', '손실', '악재', '경고', '소송', '규제', '부정적', '경쟁', '침체', '문제', '실패']
                news_sentiment_analysis = analyze_news_sentiment(news_list, positive_keywords, negative_keywords)

                # 맥스페인 계산
                max_pain_price = None
                if not option_df.empty and expiration_date:
                    price_range_for_max_pain = np.arange(current_price * 0.85, current_price * 1.15, 0.5) 
                    max_pain_price, _ = calculate_max_pain(option_df, price_range_for_max_pain)

                # 예상무브 계산
                expected_move_upper = None
                expected_move_lower = None
                if not option_df.empty and expiration_date:
                    expected_move, _ = calculate_expected_move(option_df, current_price, expiration_date, CURRENT_ANALYSIS_DATE)
                    if expected_move is not None:
                        expected_move_upper = current_price + expected_move
                        expected_move_lower = current_price - expected_move
                
                # 매매 신호 및 강도 계산 (Gemini 리포트에서 활용)
                trade_signal, buy_strength, sell_strength = generate_trade_signal(
                    current_price, max_pain_price, expected_move_lower, expected_move_upper,
                    latest_rsi, latest_sma_20, latest_sma_50, latest_sma_200, news_sentiment_analysis
                )

                # 백테스팅 실행 (Gemini 리포트에서 활용)
                backtest_results = run_backtest(
                    ticker_symbol,
                    historical_data,
                    max_pain_price_func=calculate_max_pain,
                    expected_move_func=calculate_expected_move,
                    option_data_for_backtest=option_df
                )

                with tab1: # 실시간 분석 탭
                    st.subheader("📊 실시간 분석 결과")
                    
                    st.write(f"**현재 주가:** ${current_price:.2f}")
                    st.write(f"**옵션 만기일:** {expiration_date.strftime('%Y-%m-%d') if expiration_date else '데이터 없음'}")
                    st.write(f"**맥스페인 가격:** ${max_pain_price:.2f}" if max_pain_price else "**맥스페인 가격:** 데이터 없음")
                    st.write(f"**예상무브 범위 ($1\\sigma$):** ${expected_move_lower:.2f} ~ ${expected_move_upper:.2f}" if expected_move_lower else "**예상무브 범위:** 데이터 없음")
                    st.write(f"**RSI (14일):** {latest_rsi:.2f}" if latest_rsi else "**RSI:** 데이터 없음")
                    st.write(f"**20일 이동평균선 (SMA20):** ${latest_sma_20:.2f}" if latest_sma_20 else "**SMA20:** 데이터 없음")
                    st.write(f"**50일 이동평균선 (SMA50):** ${latest_sma_50:.2f}" if latest_sma_50 else "**SMA50:** 데이터 없음")
                    st.write(f"**200일 이동평균선 (SMA200):** ${latest_sma_200:.2f}" if latest_sma_200 else "**SMA200:** 데이터 없음")
                    st.write(f"**뉴스 감성 분석:** 긍정 {news_sentiment_analysis['positive']}건, 부정 {news_sentiment_analysis['negative']}건, 중립 {news_sentiment_analysis['neutral']}건")

                    st.subheader("🤔 전략 제안")
                    if max_pain_price is not None and expected_move_lower is not None and expected_move_upper is not None and \
                       latest_rsi is not None and latest_sma_20 is not None:
                        
                        strategy_text = [] # 세부 분석 조건을 위한 리스트
                        
                        if current_price > (max_pain_price * (1 + 0.015)):
                            strategy_text.append(f"**맥스페인 상방 과매수 경고:** 현재 주가(${current_price:.2f})가 맥스페인(${max_pain_price:.2f})보다 높습니다. 만기까지 하향 수렴 가능성.")
                        elif current_price < (max_pain_price * (1 - 0.015)):
                            strategy_text.append(f"**맥스페인 하방 과매도 경고:** 현재 주가(${current_price:.2f})가 맥스페인(${max_pain_price:.2f})보다 낮습니다. 만기까지 상향 수렴 가능성.")

                        if expected_move_lower is not None and expected_move_upper is not None:
                            if current_price >= expected_move_upper * (1 - 0.005):
                                strategy_text.append(f"**예상무브 상단 근접/이탈:** 주가(${current_price:.2f})가 예상무브 상단(${expected_move_upper:.2f})에 도달. 단기 과매수 가능성.")
                            elif current_price <= expected_move_lower * (1 + 0.005):
                                strategy_text.append(f"**예상무브 하단 근접/이탈:** 주가(${current_price:.2f})가 예상무브 하단(${expected_move_lower:.2f})에 도달. 단기 과매도 가능성.")

                        if latest_rsi is not None:
                            if latest_rsi >= 70:
                                strategy_text.append(f"**RSI 과매수:** RSI({latest_rsi:.2f})가 70 이상. 단기 하락 압력 가능성.")
                            elif latest_rsi <= 30:
                                strategy_text.append(f"**RSI 과매도:** RSI({latest_rsi:.2f})가 30 이하. 단기 반등 압력 가능성.")
                        
                        if news_sentiment_analysis["positive"] > news_sentiment_analysis["negative"] * 2:
                            strategy_text.append(f"**긍정 뉴스 우세:** 긍정 뉴스 키워드가 부정 키워드보다 많음. 매수 심리 강화 가능성.")
                        elif news_sentiment_analysis["negative"] > news_sentiment_analysis["positive"] * 2:
                            strategy_text.append(f"**부정 뉴스 우세:** 부정 뉴스 키워드가 긍정 키워드보다 많음. 매도 심리 강화 가능성.")

                        st.markdown("---")
                        if trade_signal == 1:
                            st.success(f"### **✨ 강력한 매수 신호!**")
                            st.write(f"현재 가격(${current_price:.2f}) 또는 소폭 하락 시 매수 고려.")
                            st.write(f"- **손절매 (필수):** 진입 가격의 약 {(1 - 0.02)*100:.1f}% 지점 (예: ${current_price * (1 - 0.02):.2f}) 또는 예상무브 하단(${expected_move_lower:.2f}) 이탈 시.")
                            st.write(f"- **이익 실현:** 진입 가격 대비 {0.015*100:.1f}% ~ {0.03*100:.1f}% 수익 (예: ${current_price * (1 + 0.015):.2f} ~ ${current_price * (1 + 0.03):.2f}) 또는 맥스페인 근접 시.")
                        elif trade_signal == -1:
                            st.error(f"### **🔻 강력한 매도 신호!**")
                            st.write(f"현재 가격(${current_price:.2f}) 또는 소폭 상승 시 매도 고려.")
                            st.write(f"- **손절매 (필수):** 진입 가격의 약 {(1 + 0.02)*100:.1f}% 지점 (예: ${current_price * (1 + 0.02):.2f}) 또는 예상무브 상단(${expected_move_upper:.2f}) 이탈 시.")
                            st.write(f"- **이익 실현:** 진입 가격 대비 {0.015*100:.1f}% ~ {0.03*100:.1f}% 수익 (예: ${current_price * (1 - 0.015):.2f} ~ ${current_price * (1 - 0.03):.2f}) 또는 맥스페인 근접 시.")
                        else:
                            st.info("### **🟡 현재 뚜렷한 매수/매도 신호 없음 (관망).**")
                            st.write("여러 지표들이 명확한 방향을 제시하지 않고 있습니다. 신중한 접근이 필요합니다.")
                        st.markdown("---")
                        st.markdown("**세부 분석 조건:**")
                        for item in strategy_text:
                            st.markdown(f"- {item}")
                        
                        st.markdown("---")
                        st.warning("모든 투자에는 리스크가 따르며, 80% 이상의 승률은 현실적으로 매우 어렵습니다. 이 분석은 참고용입니다.")

                    else:
                        st.warning("분석을 위한 데이터가 불충분하여 전략을 제안할 수 없습니다.")
                    
                with tab2: # 백테스팅 결과 탭
                    st.subheader("📈 백테스팅 결과")
                    if backtest_results:
                        st.write(f"**초기 자본:** ${backtest_results['initial_capital']:.2f}")
                        st.write(f"**최종 포트폴리오 가치:** ${backtest_results['final_portfolio_value']:.2f}")
                        st.write(f"**총 수익률:** {backtest_results['total_return_percent']:.2f}%")
                        st.write(f"**연평균 수익률 (CAGR):** {backtest_results['cagr_percent']:.2f}%")
                        st.write(f"**승률 (Win Rate):** {backtest_results['win_rate_percent']:.2f}%")
                        st.write(f"**최대 낙폭 (Max Drawdown):** {backtest_results['max_drawdown_percent']:.2f}%")
                        st.write(f"**총 완료된 거래 횟수:** {backtest_results['total_trades']}회")
                        st.warning("⚠️ **주의:** 백테스팅 시 맥스페인/예상무브 계산에 현재 시점의 옵션 데이터가 사용됩니다. 이는 실제 과거 시장 상황을 정확히 반영하지 못하므로, 결과는 참고용으로만 활용하세요.")

                        # 포트폴리오 가치 변화 시각화
                        portfolio_df = pd.DataFrame(backtest_results['trade_log'])
                        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
                        portfolio_df.set_index('Date', inplace=True)
                        
                        fig_backtest, ax_backtest = plt.subplots(figsize=(10, 5))
                        ax_backtest.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], label='포트폴리오 가치', color='blue')
                        ax_backtest.set_title(f'{ticker_symbol} 전략 백테스팅 포트폴리오 가치 변화 ({backtest_period})')
                        ax_backtest.set_xlabel('날짜')
                        ax_backtest.set_ylabel('포트폴리오 가치 ($)')
                        ax_backtest.grid(True, linestyle='--', alpha=0.6)
                        ax_backtest.legend()
                        st.pyplot(fig_backtest)
                        plt.close(fig_backtest)
                    else:
                        st.warning("백테스팅을 실행할 수 없거나 결과가 없습니다. 과거 데이터 또는 옵션 데이터가 충분한지 확인하세요.")

                with tab3: # 분석 차트 탭
                    st.subheader("📊 통합 분석 차트")
                    if not option_df.empty and expected_move_lower is not None and expected_move_upper is not None and \
                       not historical_data_for_display.empty:
                        fig_analysis = plot_analysis_results(
                            ticker_symbol, 
                            current_price, 
                            max_pain_price, 
                            expected_move_lower, 
                            expected_move_upper, 
                            option_df,
                            historical_data_for_display 
                        )
                        st.pyplot(fig_analysis)
                        plt.close(fig_analysis)
                    else:
                        st.warning("분석 차트를 생성할 데이터가 불충분합니다 (옵션 또는 기술적 지표).")

                with tab4: # Gemini 리포트 및 추천 탭
                    st.subheader("🌟 Gemini 분석 리포트 및 추천 종목")
                    if current_price is not None and not historical_data.empty:
                        gemini_report_text = generate_gemini_report(
                            ticker_symbol, current_price, max_pain_price, expected_move_lower, expected_move_upper,
                            latest_rsi, latest_sma_20, latest_sma_50, latest_sma_200, news_sentiment_analysis,
                            backtest_results, trade_signal # trade_signal은 매매 신호 강도를 나타냄 (1, -1, 0)
                        )
                        st.markdown(gemini_report_text)
                    else:
                        st.warning("Gemini 분석 리포트를 생성할 데이터가 불충분합니다.")
