#property copyright "Copyright 2025, Damozy."
#property link      "https://www.mql5.com"
#property version   "1.02"
#property strict

#import "wininet.dll"
   int InternetOpenW(string, int, string, string, int);
   int InternetConnectW(int, string, int, string, string, int, int, int);
   int HttpOpenRequestW(int, string, string, string, string, string, int, int);
   int HttpSendRequestW(int, string, int, uchar&[], int);
   int InternetReadFile(int, uchar&[], int, int&);
   int InternetCloseHandle(int);
#import

// USER INPUTS 
input string  TradeSymbol         = "XAUUSD";      // symbol to trade
input ENUM_TIMEFRAMES EntryTF     = PERIOD_H4;     // timeframe for entries
input double  RiskPercent         = 2.0;           // % risk per trade (kept default as requested)
input double  DailyLossCapPct     = 20.0;          // % loss cap for day
input bool    UseSessionFilter    = true;
input int     SessionStartGMT     = 8;             // GMT session start
input int     SessionEndGMT       = 17;            // GMT session end

input bool    UseAIConfidence    = true;          // Enable AI confidence check
input string  AIServerURL        = "http://127.0.0.1:5000/predict"; // AI API endpoint
input double  MinAIConfidence    = 0.75;          // Minimum confidence to trade (0-1)
input bool    UseAISuggestions   = true;          // Apply AI parameter suggestions
input int     AITimeoutMS        = 3000;          // API timeout in milliseconds

// ADD THESE INCLUDES AT TOP OF FILE
//#include <WinInet.mqh>  // For HTTP requests (built-in MT4 library)

// Strategy toggles (hybrid)
input bool    UseTrendFollowing   = true;
input bool    UseMeanReversion    = false;
input bool    UseBreakout         = false;

// Strategy parameters
input int     EMA_Fast_Period     = 50;     
input int     EMA_Slow_Period     = 200;
input int     RSI_Period          = 14;

// Mean reversion / Stochastic
input int     StochK              = 14;
input int     StochD              = 3;
input int     StochSlowing        = 3;
input int     DivergenceLookback  = 5;             // lookback to check divergence (bars)
input int     FibLookback         = 100;           // fib swing lookback for breakout

// ATR / SL/TP
input int     ATRPeriod           = 14;
input double  ATRMultiplier       = 2.0;           // SL = ATR * ATRMultiplier (2x)
input double  RR_Ratio            = 3.0;           // TP = SL * RR_Ratio -> 3:1
input int     BreakEvenPipsMove   = 2;             // buffer when moving to BE (in points)

// Lots and trading limits
input int     MinLotsDecimals     = 2;
input double  MinLot              = 0.01;
input double  maxlot              = 100.0;   
input bool    AllowTrading        = true;
input int     MagicNumber         = 20250924;

// New toggles / safety parameters
input bool    CloseOpposites      = false;  // close opposite side positions before opening a new one
input bool    CloseAllOnDailyCap  = false;  // close all trades when daily loss cap hit

// Partial close & management
input bool    UsePartialClose     = true;
input double  PartialClosePct     = 50.0;

// New risk management inputs
input double  TotalDailyRiskPct      = 8.0; 
input int     MaxConcurrentTrades    = 3;     // 
input double  MaxLossPerTradeUSD    = 200.0; // 

//GLOBAL STATE 
datetime lastEntryBarTime = 0;      // last EntryTF closed bar processed
datetime lastTradeBarTime  = 0;     // per-bar execution lock
double   dayStartBalance  = 0.0;    // account balance at start of broker day
int      dayStartDate     = -1;     // date (YYYYMMDD) when dayStartBalance was stored
bool     dailyCapHit      = false;  // daily cap flag

// Position management tracking
struct PositionInfo {
    int ticket;
    bool partialClosed;
    bool movedToBE;
};
PositionInfo openPositions[100]; // Track up to 100 positions
int positionCount = 0;

// UTIL FUNCTIONS 
string SideName(int side)
{
   if(side==1) return "BUY";
   if(side==-1) return "SELL";
   return "NONE";
}

double NormalizeLotSteps(double lots)
{
   double step = MarketInfo(TradeSymbol, MODE_LOTSTEP);
   if(step <= 0) step = 0.01;
   double minlot = MarketInfo(TradeSymbol, MODE_MINLOT);
   double maxlot_broker = MarketInfo(TradeSymbol, MODE_MAXLOT);
   if(minlot <= 0) minlot = MinLot;
   // effective broker max: if broker reports 0 (unknown) use user maxlot
   if(maxlot_broker <= 0) maxlot_broker = maxlot;
   // enforce user maxlot as well
   double effectiveMax = MathMin(maxlot_broker, maxlot);

   double factor = 1.0/step;
   double normalized = MathFloor(lots * factor)/factor; 
   if(normalized < minlot) normalized = minlot;
   if(normalized > effectiveMax) normalized = effectiveMax;
   return NormalizeDouble(normalized, MinLotsDecimals);
}

double AbsPriceDiff(double a, double b)
{
   return MathAbs(a - b);
}

int ServerDayAsInt()
{
   datetime t = TimeCurrent();
   int y = TimeYear(t), m = TimeMonth(t), d = TimeDay(t);
   return y*10000 + m*100 + d;
}

// TIME & SESSION CHECKS 
bool IsWithinTradingSessionGMT()
{
   if(!UseSessionFilter) return true;
   int hr = TimeHour(TimeGMT());
   if(SessionStartGMT <= SessionEndGMT)
      return (hr >= SessionStartGMT && hr < SessionEndGMT);
   else
      return (hr >= SessionStartGMT || hr < SessionEndGMT);
}

void EnsureDayStartBalance()
{
   int today = ServerDayAsInt();
   if(dayStartDate != today)
   {
      dayStartDate = today;
      dayStartBalance = AccountBalance();
      dailyCapHit = false;
      PrintFormat("Day start reset. Date=%d Balance=%.2f", dayStartDate, dayStartBalance);
   }
}

void CheckDailyLossCap()
{
   if(DailyLossCapPct <= 0) return;
   if(dayStartBalance <= 0) return;
   double threshold = dayStartBalance * (1.0 - DailyLossCapPct/100.0);
   if(AccountBalance() <= threshold)
   {
      dailyCapHit = true;
      PrintFormat("Daily loss cap hit. StartBalance=%.2f Current=%.2f Threshold=%.2f",
                  dayStartBalance, AccountBalance(), threshold);
   }
}

void EnforceDailyCapCloseAll()
{
   if(!dailyCapHit || !CloseAllOnDailyCap) return;
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != TradeSymbol) continue;
      if(OrderMagicNumber() != MagicNumber) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;

      double closePrice = (OrderType() == OP_BUY) ? MarketInfo(TradeSymbol, MODE_BID) : MarketInfo(TradeSymbol, MODE_ASK);
      bool ok = OrderClose(OrderTicket(), OrderLots(), closePrice, 8, clrRed);
      if(!ok) {
         int err = GetLastError();
         PrintFormat("EnforceDailyCapCloseAll: failed to close ticket=%d err=%d", OrderTicket(), err);
         ResetLastError();
      } else {
         PrintFormat("EnforceDailyCapCloseAll: closed ticket=%d", OrderTicket());
      }
   }
}

//ATR / SL/TP / LOT
double GetATR(int period, int shift, ENUM_TIMEFRAMES tf)
{
   double atr = iATR(TradeSymbol, tf, period, shift);
   return atr;
}

double ComputeSLPriceFromATR(int side, ENUM_TIMEFRAMES tf)
{
   // returns distance in price units (absolute) to be used for SL
   double atr = GetATR(ATRPeriod, 1, tf);
   if(atr <= 0) {
      Print("ComputeSLPriceFromATR: ATR invalid");
      return 0; // Safety check
   }
   return atr * ATRMultiplier;
}

// Original LotsForRisk retained (kept for compatibility)
double LotsForRisk(double stopDistancePrice, double riskPercent)
{
   // === Safety: Minimum lot
   if(stopDistancePrice <= 0.0)
      return NormalizeLotSteps(MinLot);

   // --- Risk in account currency
   double riskAmount = AccountEquity() * (riskPercent / 100.0);
   if(riskAmount <= 0) return NormalizeLotSteps(MinLot);

   // --- Market info
   double point     = MarketInfo(TradeSymbol, MODE_POINT);
   double tickValue = MarketInfo(TradeSymbol, MODE_TICKVALUE);
   double stopLevel = MarketInfo(TradeSymbol, MODE_STOPLEVEL) * point;

   // Fallbacks for tickValue & required margin
   if(point <= 0) point = 0.01;
   if(tickValue <= 0) {
      // best-effort fallback: assume tick value equals 10 * point (conservative for XAU variants)
      tickValue = point * 10.0;
      PrintFormat("LotsForRisk: tickValue fallback used=%.5f", tickValue);
   }

   double requiredMarginPerLot = MarketInfo(TradeSymbol, MODE_MARGINREQUIRED);
   if(requiredMarginPerLot <= 0) {
      // fallback: estimate margin per lot as ~1% of account balance (conservative)
      requiredMarginPerLot = MathMax(AccountBalance() * 0.01, point * 1000.0);
      PrintFormat("LotsForRisk: marginRequired fallback used=%.2f", requiredMarginPerLot);
   }

   // --- Enforce a minimum stop distance (to avoid huge lots on tiny stops)
   double minStop = stopLevel * 2; // 2x broker minimum stop level
   if(minStop <= 0) minStop = point * 10; // fallback
   if(stopDistancePrice < minStop) stopDistancePrice = minStop;

   // --- Convert stop to pips (points)
   double stopPips = stopDistancePrice / point;

   // --- Loss per lot
   double lossPerLot = stopPips * tickValue;
   if(lossPerLot <= 0) return NormalizeLotSteps(MinLot);

   // --- Raw lots
   double rawLots = riskAmount / lossPerLot;
   if(rawLots <= 0) return NormalizeLotSteps(MinLot);

   // --- Margin check (80% safety)
   double freeMargin           = AccountFreeMargin();
   if(requiredMarginPerLot > 0 && freeMargin > 0) {
      double maxByMargin = (freeMargin / requiredMarginPerLot) * 0.8;
      if(rawLots > maxByMargin) {
         rawLots = maxByMargin;
         PrintFormat("Lot size capped by margin: %.2f", rawLots);
      }
   }

   // --- Cap swings between trades (stability)
   static double lastLot = 0;
   if(lastLot > 0) {
      if(rawLots > lastLot * 2.0) rawLots = lastLot * 2.0;   // limit sudden growth
      if(rawLots < lastLot * 0.5) rawLots = lastLot * 0.5;   // limit sudden shrink
   }
   lastLot = rawLots;

   // --- Normalize with broker rules
   double lots = NormalizeLotSteps(rawLots);

   // --- Apply broker min/max
   double brokerMin = MarketInfo(TradeSymbol, MODE_MINLOT);
   if(brokerMin <= 0) brokerMin = MinLot;
   double brokerMax = MarketInfo(TradeSymbol, MODE_MAXLOT);
   if(brokerMax <= 0) brokerMax = maxlot;
   double effectiveMax = MathMin(brokerMax, maxlot);

   if(lots < brokerMin) lots = brokerMin;
   if(lots > effectiveMax) lots = effectiveMax;

   // --- Debug log (very important)
   PrintFormat("LotsForRisk: Equity=%.2f, Risk=%.2f%%, StopDist=%.5f, StopPips=%.1f, LossPerLot=%.2f, RawLots=%.2f, FinalLots=%.2f",
               AccountEquity(), riskPercent, stopDistancePrice, stopPips, lossPerLot, rawLots, lots);

   return NormalizeLotSteps(lots);
}

// SAFER LOT CALC: LOTS FOR DOLLAR RISK 

double LotsForDollarRisk(double stopDistancePrice, double dollarRiskAllowed)
{
   if(stopDistancePrice <= 0 || dollarRiskAllowed <= 0) return NormalizeLotSteps(MinLot);

   double point = MarketInfo(TradeSymbol, MODE_POINT); if(point <= 0) point = 0.01;
   double tickValue = MarketInfo(TradeSymbol, MODE_TICKVALUE);
   if(tickValue <= 0) { tickValue = point * 10.0; PrintFormat("LotsForDollarRisk: tickValue fallback used=%.5f", tickValue); }

   double stopPips = stopDistancePrice / point;
   double lossPerLot = stopPips * tickValue;
   if(lossPerLot <= 0) return NormalizeLotSteps(MinLot);

   double rawLots = dollarRiskAllowed / lossPerLot;

   // Margin cap like current function:
   double requiredMarginPerLot = MarketInfo(TradeSymbol, MODE_MARGINREQUIRED);
   if(requiredMarginPerLot <= 0) requiredMarginPerLot = MathMax(AccountBalance() * 0.01, point * 1000.0);
   double freeMargin = AccountFreeMargin();
   if(requiredMarginPerLot > 0 && freeMargin > 0) {
      double maxByMargin = (freeMargin / requiredMarginPerLot) * 0.8;
      if(rawLots > maxByMargin) rawLots = maxByMargin;
   }

   // apply smoothing / lastLot protection
   static double lastSafeLot = 0;
   if(lastSafeLot > 0) {
      if(rawLots > lastSafeLot * 2.0) rawLots = lastSafeLot * 2.0;
      if(rawLots < lastSafeLot * 0.5) rawLots = lastSafeLot * 0.5;
   }
   lastSafeLot = rawLots;

   double lots = NormalizeLotSteps(rawLots);
   double brokerMin = MarketInfo(TradeSymbol, MODE_MINLOT);
   if(brokerMin <= 0) brokerMin = MinLot;
   double brokerMax = MarketInfo(TradeSymbol, MODE_MAXLOT);
   if(brokerMax <= 0) brokerMax = maxlot;
   double effectiveMax = MathMin(brokerMax, maxlot);
   if(lots < brokerMin) lots = brokerMin;
   if(lots > effectiveMax) lots = effectiveMax;
   return NormalizeLotSteps(lots);
}

//POSITION TRACKING FUNCTIONS 
int FindPositionIndex(int ticket) {
   for(int i = 0; i < positionCount; i++) {
      if(openPositions[i].ticket == ticket) return i;
   }
   return -1;
}

void AddPosition(int ticket) {
   if(positionCount >= 100) return; // Array full
   openPositions[positionCount].ticket = ticket;
   openPositions[positionCount].partialClosed = false;
   openPositions[positionCount].movedToBE = false;
   positionCount++;
}

void RemovePosition(int ticket) {
   int index = FindPositionIndex(ticket);
   if(index < 0) return;
   for(int i = index; i < positionCount - 1; i++) {
      openPositions[i] = openPositions[i + 1];
   }
   positionCount--;
}

//STRATEGY SIGNALS 

// 1) Trend-Following: EMA50/200 crossover on closed EntryTF + RSI confirmation
int Signal_TrendFollowing()
{
   // Check if we have enough bars
   if(iBars(TradeSymbol, EntryTF) < MathMax(EMA_Fast_Period, EMA_Slow_Period) + 5) {
      //Print("Not enough bars for trend following signal");
      return 0;
   }

   // Evaluate on closed candle (shift 1)
   double emaFast_prev = iMA(TradeSymbol, EntryTF, EMA_Fast_Period, 0, MODE_EMA, PRICE_CLOSE, 2);
   double emaSlow_prev = iMA(TradeSymbol, EntryTF, EMA_Slow_Period, 0, MODE_EMA, PRICE_CLOSE, 2);
   double emaFast_now  = iMA(TradeSymbol, EntryTF, EMA_Fast_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   double emaSlow_now  = iMA(TradeSymbol, EntryTF, EMA_Slow_Period, 0, MODE_EMA, PRICE_CLOSE, 1);

   double rsi_now = iRSI(TradeSymbol, EntryTF, RSI_Period, PRICE_CLOSE, 1);

   // Validate indicator values
   if(emaFast_prev <= 0 || emaSlow_prev <= 0 || emaFast_now <= 0 || emaSlow_now <= 0 || rsi_now <= 0) {
      return 0;
   }

   // Crossover detection on closed bar
   bool buyCross = (emaFast_prev <= emaSlow_prev && emaFast_now > emaSlow_now); //bulish signal
   bool sellCross= (emaFast_prev >= emaSlow_prev && emaFast_now < emaSlow_now); // bearish signal

   if(buyCross && rsi_now > 50.0) return 1;
   if(sellCross && rsi_now < 50.0) return -1;
   return 0;
}

// 2) Mean Reversion: RSI divergence + Stochastic confirmation
int Signal_MeanReversion()
{
  if(iBars(TradeSymbol, EntryTF) < RSI_Period + DivergenceLookback + 5) {
      return 0;
   }
   
   int lb = MathMax(3, DivergenceLookback);
   int swingBarsBack = 5;
   
   double high1 = -1, high2 = -1, low1 = 1e12, low2 = 1e12;
   double rsi1_h = 0, rsi2_h = 0, rsi1_l = 0, rsi2_l = 0;
   
   for(int i = 2; i <= swingBarsBack + 2; i++) {
      double h = iHigh(TradeSymbol, EntryTF, i);
      double l = iLow(TradeSymbol, EntryTF, i);
      double rsi_h = iRSI(TradeSymbol, EntryTF, RSI_Period, PRICE_CLOSE, i);
      double rsi_l = iRSI(TradeSymbol, EntryTF, RSI_Period, PRICE_CLOSE, i);

      if(h > high1) {
         high2 = high1; rsi2_h = rsi1_h;
         high1 = h; rsi1_h = rsi_h;
      } else if(h > high2) {
         high2 = h; rsi2_h = rsi_h;
      }

      if(l < low1) {
         low2 = low1; rsi2_l = rsi1_l;
         low1 = l; rsi1_l = rsi_l;
      } else if(l < low2) {
         low2 = l; rsi2_l = rsi_l;
      }
   }

   double k_now = iStochastic(TradeSymbol, EntryTF, StochK, StochD, StochSlowing, MODE_SMA, STO_LOWHIGH, MODE_MAIN, 1);
   double d_now = iStochastic(TradeSymbol, EntryTF, StochK, StochD, StochSlowing, MODE_SMA, STO_LOWHIGH, MODE_SIGNAL, 1);
   double k_prev= iStochastic(TradeSymbol, EntryTF, StochK, StochD, StochSlowing, MODE_SMA, STO_LOWHIGH, MODE_MAIN, 2);
   double d_prev= iStochastic(TradeSymbol, EntryTF, StochK, StochD, StochSlowing, MODE_SMA, STO_LOWHIGH, MODE_SIGNAL, 2);

   if(k_now < 0 || d_now < 0 || k_prev < 0 || d_prev < 0) return 0;

   bool priceLowerLow    = (low1 < low2) && (low2 < 1e12);
   bool rsiHigherLow     = (rsi1_l > rsi2_l) && (rsi1_l > 0) && (rsi2_l > 0);
   bool rsiNearOversold  = (rsi1_l < 40.0);
   bool stochCrossUp = (k_prev <= d_prev && k_now > d_now && k_prev < 30.0);
   
   if(priceLowerLow && rsiHigherLow && rsiNearOversold && stochCrossUp)
      return 1;
   
   bool priceHigherHigh  = (high1 > high2) && (high1 > 0);
   bool rsiLowerHigh     = (rsi1_h < rsi2_h) && (rsi1_h > 0) && (rsi2_h > 0);
   bool rsiNearOverbought= (rsi1_h > 60.0);
   bool stochCrossDown   = (k_prev >= d_prev && k_now < d_prev && k_now > 70.0);
   
   if(priceHigherHigh && rsiLowerHigh && rsiNearOverbought && stochCrossDown)
      return -1;

   return 0;
}

// 3) Breakout: Bollinger Bands close beyond band + volume confirmation
int Signal_Breakout()
{
   if(iBars(TradeSymbol, EntryTF) < FibLookback + 25) {
      return 0;
   }

   int bb_period = 20;
   double bb_devs = 2.0;
   double upper = iBands(TradeSymbol, EntryTF, bb_period, bb_devs, 0, PRICE_CLOSE, MODE_UPPER, 1);
   double lower = iBands(TradeSymbol, EntryTF, bb_period, bb_devs, 0, PRICE_CLOSE, MODE_LOWER, 1);
   double close_now = iClose(TradeSymbol, EntryTF, 1);
   
   double vol_now = iVolume(TradeSymbol, EntryTF, 1);
   double vol_avg = 0;
   int vol_bars = MathMin(10, iBars(TradeSymbol, EntryTF) - 2);
   if(vol_bars < 5) return 0;
   for(int i = 2; i <= vol_bars + 1; i++) {
      vol_avg += iVolume(TradeSymbol, EntryTF, i);
   }
   vol_avg /= vol_bars;

   if(upper <= 0 || lower <= 0 || upper <= lower) return 0;
   if(close_now <= 0 || vol_now <= 0 || vol_avg <= 0) return 0;

   int look = MathMax(20, FibLookback);
   int idxHigh = iHighest(TradeSymbol, EntryTF, MODE_HIGH, look, 1);
   int idxLow  = iLowest(TradeSymbol, EntryTF, MODE_LOW, look, 1);
   if(idxHigh < 0 || idxLow < 0) return 0;
   double swingHigh = iHigh(TradeSymbol, EntryTF, idxHigh);
   double swingLow  = iLow(TradeSymbol, EntryTF, idxLow);
   double range = swingHigh - swingLow;
   if(range <= 0) return 0;
   
   double fib618_fromLow = swingLow + range * 0.618;
   double fib382_fromHigh = swingHigh - range * 0.382;

   bool volSpike = vol_now > vol_avg * 1.5;

   if(close_now > upper && volSpike && close_now > fib618_fromLow)
      return 1;

   if(close_now < lower && volSpike && close_now < fib382_fromHigh)
      return -1;

   return 0;
}

//TRADE SEND / MANAGEMENT 
int trade_SendOrder(int magic, int side, double lots, double sl, double tp)
{
   if(lots <= 0) { Print("trade_SendOrder: invalid lots"); return -1; }

   double minLot = MarketInfo(TradeSymbol, MODE_MINLOT);
   double maxLot_broker = MarketInfo(TradeSymbol, MODE_MAXLOT);
   if(maxLot_broker <= 0) maxLot_broker = maxlot;
   if(minLot <= 0) minLot = MinLot;
   
   lots = NormalizeLotSteps(lots);
   if(lots < minLot) {
      PrintFormat("Lot size %.2f below minimum %.2f", lots, minLot);
      return -1;
   }
   if(lots > MathMin(maxLot_broker, maxlot)) {
      lots = MathMin(maxLot_broker, maxlot);
   }

   // Check margin requirement
   double requiredMargin = MarketInfo(TradeSymbol, MODE_MARGINREQUIRED) * lots;
   double freeMargin = AccountFreeMargin();
   if(requiredMargin > 0 && requiredMargin > freeMargin * 0.8) {
      Print("Insufficient margin for trade");
      return -1;
   }
   
   RefreshRates(); // ensure Ask/Bid are up-to-date
   double price = (side == 1) ? MarketInfo(TradeSymbol, MODE_ASK) : MarketInfo(TradeSymbol, MODE_BID);
   double spread = MarketInfo(TradeSymbol, MODE_SPREAD) * MarketInfo(TradeSymbol, MODE_POINT);
   double point = MarketInfo(TradeSymbol, MODE_POINT);

   int slippage = MathMin(50, MathMax(3, (int)(spread * 3 / point))); // capped slippage
   int type = (side==1) ? OP_BUY : OP_SELL;

   // Validate SL and TP levels
   double minStopLevel = MarketInfo(TradeSymbol, MODE_STOPLEVEL) * point;
   if(minStopLevel <= 0) minStopLevel = point * 10; // fallback
   if(side == 1) {
      if(sl > 0 && (price - sl) < minStopLevel) {
         PrintFormat("Stop loss too close to price. Min distance: %.5f, Current: %.5f", minStopLevel, price - sl);
         sl = price - minStopLevel;
      }
      if(tp > 0 && (tp - price) < minStopLevel) {
         tp = price + minStopLevel;
      }
   } else {
      if(sl > 0 && (sl - price) < minStopLevel) {
         sl = price + minStopLevel;
      }
      if(tp > 0 && (price - tp) < minStopLevel) {
         tp = price - minStopLevel;
      }
   }

   int ticket = OrderSend(TradeSymbol, type, lots, price, slippage, sl, tp, "GoldBot", magic, 0, clrBlue);
   if(ticket < 0)
   {
      int err = GetLastError();
      PrintFormat("OrderSend failed: side=%s lots=%.2f price=%.5f sl=%.5f tp=%.5f err=%d", 
                  SideName(side), lots, price, sl, tp, err);
      ResetLastError();
   }
   else
   {
      AddPosition(ticket); // Track the position
      PrintFormat("OrderSend success: ticket=%d side=%s lots=%.2f SL=%.5f TP=%.5f",
                  ticket, SideName(side), lots, sl, tp);
   }
   return ticket;
}

void CloseOppositePositions(int oppositeSide)
{
   if(!CloseOpposites) return; // respect user toggle

   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != TradeSymbol) continue;
      if(OrderMagicNumber() != MagicNumber) continue;

      if(oppositeSide == 1 && OrderType() == OP_SELL)
      {
         double closeLots = OrderLots();
         RefreshRates(); // ensure Ask/Bid are up-to-date
         double closePrice = MarketInfo(TradeSymbol, MODE_ASK);
         bool ok = OrderClose(OrderTicket(), closeLots, closePrice, 8, clrRed);
         if(!ok) {
            int err = GetLastError();
            PrintFormat("CloseOppositePositions: failed to close sell ticket=%d err=%d", OrderTicket(), err);
            ResetLastError();
         } else {
            RemovePosition(OrderTicket()); // Remove from tracking
            PrintFormat("Closed sell ticket=%d", OrderTicket());
         }
      }
      else if(oppositeSide == -1 && OrderType() == OP_BUY)
      {
         double closeLots = OrderLots();
         double closePrice = MarketInfo(TradeSymbol, MODE_BID);
         bool ok = OrderClose(OrderTicket(), closeLots, closePrice, 8, clrRed);
         if(!ok) {
            int err = GetLastError();
            PrintFormat("CloseOppositePositions: failed to close buy ticket=%d err=%d", OrderTicket(), err);
         } else {
            RemovePosition(OrderTicket()); // Remove from tracking
            PrintFormat("Closed buy ticket=%d", OrderTicket());
         }
      }
   }
}

void ManageOpenPositions()
{
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != TradeSymbol) continue;
      if(OrderMagicNumber() != MagicNumber) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;

      int posIndex = FindPositionIndex(OrderTicket());
      if(posIndex < 0) {
         AddPosition(OrderTicket()); // Add if not tracked
         posIndex = positionCount - 1;
      }

      double openPrice = OrderOpenPrice();
      double currentPrice = (OrderType() == OP_BUY) ? 
                              MarketInfo(TradeSymbol, MODE_BID) : 
                              MarketInfo(TradeSymbol, MODE_ASK);
      double sl = OrderStopLoss();
      double tp = OrderTakeProfit();
      double stopDistance = AbsPriceDiff(openPrice, sl);

      if(stopDistance <= 0) continue; // Skip if no valid SL

      double profitPriceMove = (OrderType() == OP_BUY) ? 
                                 (currentPrice - openPrice) : 
                                 (openPrice - currentPrice);
      double riskRewardRatio = profitPriceMove / stopDistance;

      // Breakeven Logic 
      if(riskRewardRatio >= 1.0 && !openPositions[posIndex].movedToBE)
      {
         double newSL;
         double beBuffer = BreakEvenPipsMove * MarketInfo(TradeSymbol, MODE_POINT);
         if(OrderType() == OP_BUY)
            newSL = openPrice + beBuffer;
         else
            newSL = openPrice - beBuffer;

         int digits = (int)MarketInfo(TradeSymbol, MODE_DIGITS);
         newSL = NormalizeDouble(newSL, digits);

         bool needModify = false;
         double point = MarketInfo(TradeSymbol, MODE_POINT);
         if(point <= 0) point = 0.01;
         if(OrderType() == OP_BUY && (sl == 0 || sl < newSL - point)) needModify = true;
         if(OrderType() == OP_SELL && (sl == 0 || sl > newSL + point)) needModify = true;

         if(needModify)
         {
            if(!OrderModify(OrderTicket(), openPrice, newSL, tp, 0, clrYellow))
            {
               int err = GetLastError();
               PrintFormat("OrderModify(BE) failed ticket=%d err=%d", OrderTicket(), err);
               ResetLastError();
            } 
            else 
            {
               openPositions[posIndex].movedToBE = true;
               PrintFormat("Moved SL to BE for ticket=%d newSL=%.5f", OrderTicket(), newSL);
            }
         }
      }

      //  Partial Close Logic 
      if(UsePartialClose && riskRewardRatio >= 1.0 && !openPositions[posIndex].partialClosed)
      {
         double currentLots = OrderLots();
         double targetCloseLots = NormalizeLotSteps(currentLots * (PartialClosePct / 100.0));
         double minLot = MarketInfo(TradeSymbol, MODE_MINLOT);

         double remainingLots = currentLots - targetCloseLots;
         if(targetCloseLots >= minLot && remainingLots >= minLot)
         {
            double closePrice = (OrderType() == OP_BUY) ? 
                                  MarketInfo(TradeSymbol, MODE_BID) : 
                                  MarketInfo(TradeSymbol, MODE_ASK);
            bool closed = OrderClose(OrderTicket(), targetCloseLots, closePrice, 8, clrMagenta);

            if(closed) {
               openPositions[posIndex].partialClosed = true;
               PrintFormat("Partial close executed ticket=%d closeLots=%.2f remaining=%.2f", 
                          OrderTicket(), targetCloseLots, remainingLots);
            } else {
               int err = GetLastError();
               PrintFormat("Partial close failed ticket=%d err=%d", OrderTicket(), err);
               ResetLastError();
            }
         }
      }
   }

   //  Cleanup Closed Positions 
   for(int i = positionCount - 1; i >= 0; i--) {
      bool found = false;
      for(int j = OrdersTotal() - 1; j >= 0; j--) {
         if(!OrderSelect(j, SELECT_BY_POS, MODE_TRADES)) continue;
         if(OrderTicket() == openPositions[i].ticket) {
            found = true;
            break;
         }
      }
      if(!found) {
         PrintFormat("Removing closed/expired position ticket=%d", openPositions[i].ticket);
         RemovePosition(openPositions[i].ticket);
      }
   }
}

// ON TICK 
bool CanTradeNow()
{
   datetime barTime = iTime(TradeSymbol, EntryTF, 1);
   if(barTime <= 0) return false;
   if(barTime == lastTradeBarTime) return false;
   lastTradeBarTime = barTime;
   return true;
}

// ============================================================================
// AI CONFIDENCE CHECKING FUNCTIONS
// ============================================================================

string PrepareAIRequestJSON(int strategy, int signal)
{
   // Gather current market conditions and indicators
   double emaFast = iMA(TradeSymbol, EntryTF, EMA_Fast_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   double emaSlow = iMA(TradeSymbol, EntryTF, EMA_Slow_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   double rsi = iRSI(TradeSymbol, EntryTF, RSI_Period, PRICE_CLOSE, 1);
   double atr = GetATR(ATRPeriod, 1, EntryTF);
   double stochK = iStochastic(TradeSymbol, EntryTF, StochK, StochD, StochSlowing, MODE_SMA, STO_LOWHIGH, MODE_MAIN, 1);
   double stochD = iStochastic(TradeSymbol, EntryTF, StochK, StochD, StochSlowing, MODE_SMA, STO_LOWHIGH, MODE_SIGNAL, 1);
   
   double volume_current = iVolume(TradeSymbol, EntryTF, 1);
   double volume_avg = 0;
   for(int i = 2; i <= 11; i++) volume_avg += iVolume(TradeSymbol, EntryTF, i);
   volume_avg /= 10.0;
   double volumeRatio = (volume_avg > 0) ? volume_current / volume_avg : 1.0;
   
   double spread = MarketInfo(TradeSymbol, MODE_SPREAD) * MarketInfo(TradeSymbol, MODE_POINT);
   int hour = TimeHour(TimeGMT());
   
   double close_price = iClose(TradeSymbol, EntryTF, 1);
   double bb_upper = iBands(TradeSymbol, EntryTF, 20, 2, 0, PRICE_CLOSE, MODE_UPPER, 1);
   double bb_lower = iBands(TradeSymbol, EntryTF, 20, 2, 0, PRICE_CLOSE, MODE_LOWER, 1);
   
   // Calculate derived features
   double emaDistance = (emaSlow > 0) ? ((emaFast - emaSlow) / emaSlow * 100) : 0;
   double atrPct = (close_price > 0) ? (atr / close_price * 100) : 0;
   
   string strategyName = "unknown";
   if(strategy == 0) strategyName = "trend_following";
   else if(strategy == 1) strategyName = "mean_reversion";
   else if(strategy == 2) strategyName = "breakout";
   
   // Build JSON string (note: MQL4 doesn't have native JSON, so we build it manually)
   string json = "{";
   json += "\"ema_fast\":" + DoubleToString(emaFast, 2) + ",";
   json += "\"ema_slow\":" + DoubleToString(emaSlow, 2) + ",";
   json += "\"rsi\":" + DoubleToString(rsi, 2) + ",";
   json += "\"atr\":" + DoubleToString(atr, 2) + ",";
   json += "\"stoch_k\":" + DoubleToString(stochK, 2) + ",";
   json += "\"stoch_d\":" + DoubleToString(stochD, 2) + ",";
   json += "\"bb_upper\":" + DoubleToString(bb_upper, 2) + ",";
   json += "\"bb_lower\":" + DoubleToString(bb_lower, 2) + ",";
   json += "\"volume_ratio\":" + DoubleToString(volumeRatio, 2) + ",";
   json += "\"spread\":" + DoubleToString(spread, 5) + ",";
   json += "\"hour_of_day\":" + IntegerToString(hour) + ",";
   json += "\"ema_distance\":" + DoubleToString(emaDistance, 2) + ",";
   json += "\"atr_pct\":" + DoubleToString(atrPct, 4) + ",";
   json += "\"signal\":" + IntegerToString(signal) + ",";
   json += "\"strategy_type\":\"" + strategyName + "\",";
   json += "\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"";
   json += "}";
   
   return json;
}

// Simplified HTTP POST function for MT4
string SendHTTPRequest(string url, string jsonData)
{
   string headers = "Content-Type: application/json\r\n";
   string result = "";
   
   // Use WinInet functions to make HTTP request
   int hInternet = InternetOpenW("MT4 EA Client", 1, "", "", 0);
   if(hInternet == 0)
   {
      Print("AI HTTP: InternetOpenW failed");
      return "";
   }
   
   string server = "127.0.0.1";
   int port = 5000;
   string path = "/predict";
   
   int hConnect = InternetConnectW(hInternet, server, port, "", "", 3, 0, 0);
   if(hConnect == 0)
   {
      InternetCloseHandle(hInternet);
      Print("AI HTTP: InternetConnectW failed");
      return "";
   }
   
   int hRequest = HttpOpenRequestW(hConnect, "POST", path, "HTTP/1.1", "", "", 0, 0);
   if(hRequest == 0)
   {
      InternetCloseHandle(hConnect);
      InternetCloseHandle(hInternet);
      Print("AI HTTP: HttpOpenRequestW failed");
      return "";
   }
   
   uchar data[];
   StringToCharArray(jsonData, data, 0, StringLen(jsonData));
   
   if(HttpSendRequestW(hRequest, headers, StringLen(headers), data, ArraySize(data)))
   {
      char buffer[4096];
      uint bytes = 0;
      
      if(InternetReadFile(hRequest, buffer, 4096, bytes))
      {
         result = CharArrayToString(buffer, 0, bytes);
      }
   }
   
   InternetCloseHandle(hRequest);
   InternetCloseHandle(hConnect);
   InternetCloseHandle(hInternet);
   
   return result;
}

// Parse JSON response (simplified - looks for key values)
double ExtractJSONDouble(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int pos = StringFind(json, searchKey);
   if(pos < 0) return -999.0; // Not found
   
   int startPos = pos + StringLen(searchKey);
   int endPos = startPos;
   
   // Find end of number (comma, } or ])
   while(endPos < StringLen(json))
   {
      string c = StringSubstr(json, endPos, 1);
      if(c == "," || c == "}" || c == "]") break;
      endPos++;
   }
   
   string valueStr = StringSubstr(json, startPos, endPos - startPos);
   StringTrimLeft(valueStr);
   StringTrimRight(valueStr);
   
   return StringToDouble(valueStr);
}

bool ExtractJSONBool(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int pos = StringFind(json, searchKey);
   if(pos < 0) return false;
   
   int startPos = pos + StringLen(searchKey);
   string snippet = StringSubstr(json, startPos, 10);
   
   if(StringFind(snippet, "true") >= 0) return true;
   return false;
}

string ExtractJSONString(string json, string key)
{
   string searchKey = "\"" + key + "\":\"";
   int pos = StringFind(json, searchKey);
   if(pos < 0) return "";
   
   int startPos = pos + StringLen(searchKey);
   int endPos = StringFind(json, "\"", startPos);
   
   if(endPos < 0) return "";
   
   return StringSubstr(json, startPos, endPos - startPos);
}

// Main AI confidence check function
bool CheckAIConfidence(int strategy, int signal, double &confidenceScore, string &suggestions)
{
   if(!UseAIConfidence)
   {
      confidenceScore = 1.0; // Assume full confidence if AI disabled
      return true;
   }
   
   PrintFormat("[AI] Checking confidence for strategy=%d signal=%d", strategy, signal);
   
   // Prepare request
   string jsonRequest = PrepareAIRequestJSON(strategy, signal);
   PrintFormat("[AI] Request: %s", jsonRequest);
   
   // Send HTTP request
   string response = SendHTTPRequest(AIServerURL, jsonRequest);
   
   if(StringLen(response) == 0)
   {
      Print("[AI] WARNING: No response from AI server - defaulting to ALLOW trade");
      confidenceScore = MinAIConfidence; // Fail-safe: allow trade if server down
      return true;
   }
   
   PrintFormat("[AI] Response: %s", response);
   
   // Parse response
   confidenceScore = ExtractJSONDouble(response, "confidence");
   bool shouldTrade = ExtractJSONBool(response, "should_trade");
   string confidenceLevel = ExtractJSONString(response, "confidence_level");
   
   // Extract suggestions if present
   int suggPos = StringFind(response, "\"suggestions\":");
   if(suggPos >= 0)
   {
      string suggSection = StringSubstr(response, suggPos);
      suggestions = suggSection; // Store full suggestions section
   }
   
   PrintFormat("[AI] Confidence=%.2f Level=%s ShouldTrade=%s", 
               confidenceScore, confidenceLevel, shouldTrade ? "YES" : "NO");
   
   // Log suggestions if confidence is low
   if(!shouldTrade && StringLen(suggestions) > 0)
   {
      string waitReason = ExtractJSONString(response, "wait_reason");
      string action = ExtractJSONString(response, "action");
      
      if(StringLen(waitReason) > 0)
         PrintFormat("[AI] Wait Reason: %s", waitReason);
      if(StringLen(action) > 0)
         PrintFormat("[AI] Suggested Action: %s", action);
   }
   
   return shouldTrade;
}


void ApplyAISuggestions(string suggestions, double &atrMult, double &rrRatio, double &lots)
{
   if(!UseAISuggestions || StringLen(suggestions) == 0) return;
   
   // Extract suggested parameter adjustments
   double suggestedATRMult = ExtractJSONDouble(suggestions, "suggested_atr_multiplier");
   if(suggestedATRMult > 0 && suggestedATRMult != -999.0)
   {
      PrintFormat("[AI] Applying ATR multiplier suggestion: %.2f -> %.2f", atrMult, suggestedATRMult);
      atrMult = suggestedATRMult;
   }
   
   double lotReduction = ExtractJSONDouble(suggestions, "suggested_lot_reduction");
   if(lotReduction > 0 && lotReduction <= 1.0 && lotReduction != -999.0)
   {
      PrintFormat("[AI] Applying lot reduction: %.2f -> %.2f", lots, lots * lotReduction);
      lots = lots * lotReduction;
      lots = NormalizeLotSteps(lots);
   }
}

void OnTick()
{
   if(!AllowTrading) return;
   if(Symbol() != TradeSymbol) return;

   EnsureDayStartBalance();
   CheckDailyLossCap();
   if(dailyCapHit)
   {
      Print("Daily loss cap hit - trading suspended for the day.");
      EnforceDailyCapCloseAll();
      return;
   }

   // Session filter
   if(!IsWithinTradingSessionGMT())
   {
      ManageOpenPositions();
      return;
   }

   // Manage existing positions (BE/partial)
   ManageOpenPositions();

   // Only act on new closed candle on EntryTF
   datetime closedBar = iTime(TradeSymbol, EntryTF, 1);
   if(closedBar <= lastEntryBarTime) return; // Changed from == to <= for safety
   lastEntryBarTime = closedBar;

   // prevent duplicate trade execution on same bar
   if(!CanTradeNow()) return;

   // Check if we have enough historical data
   if(iBars(TradeSymbol, EntryTF) < MathMax(EMA_Slow_Period, FibLookback) + 10) {
      Print("Insufficient historical data for analysis");
      return;
   }

   int sigTrend = 0, sigMean = 0, sigBreak = 0;
   if(UseTrendFollowing) sigTrend = Signal_TrendFollowing();
   if(UseMeanReversion) sigMean  = Signal_MeanReversion();
   if(UseBreakout)      sigBreak = Signal_Breakout();

   // Count current open lots for this EA+symbol to enforce maxlot limit
   double currentTotalLots = 0;
   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != TradeSymbol) continue;
      if(OrderMagicNumber() != MagicNumber) continue;
      if(OrderType() == OP_BUY || OrderType() == OP_SELL) {
         currentTotalLots += OrderLots();
      }
   }

   double remainingLotsAllowed = maxlot - currentTotalLots;
   double minLot = MarketInfo(TradeSymbol, MODE_MINLOT);
   if(minLot <= 0) minLot = MinLot;
   
   if(remainingLotsAllowed < minLot)
   {
      Print("No remaining lot allowance (maxlot reached) - skipping new entries this bar.");
      return;
   }

   int signals[3]; 
   signals[0] = sigTrend; 
   signals[1] = sigMean; 
   signals[2] = sigBreak;
   
   string stratNames[3]; 
   stratNames[0] = "TrendFollowing"; 
   stratNames[1] = "MeanReversion"; 
   stratNames[2] = "Breakout";

   for(int s = 0; s < 3; s++)
   {
      int sig = signals[s];
      if(sig == 0) continue;

      if(remainingLotsAllowed < minLot) {
         PrintFormat("[%s] No remaining lot capacity", stratNames[s]);
         break;
      }

      // Close opposite positions before opening new trade (toggleable)
      if(CloseOpposites) {
         if(sig == 1) CloseOppositePositions(1);
         else if(sig == -1) CloseOppositePositions(-1);
      }

      // Recalculate remaining lots after closing opposite positions
      currentTotalLots = 0;
      for(int i = 0; i < OrdersTotal(); i++)
      {
         if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
         if(OrderSymbol() != TradeSymbol) continue;
         if(OrderMagicNumber() != MagicNumber) continue;
         if(OrderType() == OP_BUY || OrderType() == OP_SELL) {
            currentTotalLots += OrderLots();
         }
      }
      remainingLotsAllowed = maxlot - currentTotalLots;
      
      // ==============================================
      // AI CONFIDENCE CHECK - ADD THIS SECTION
      // ==============================================
      double aiConfidence = 0;
      string aiSuggestions = "";
      bool aiApproved = CheckAIConfidence(s, sig, aiConfidence, aiSuggestions);
      
      if(!aiApproved)
      {
         PrintFormat("[%s] AI REJECTED trade - Confidence %.2f < threshold %.2f", 
                     stratNames[s], aiConfidence, MinAIConfidence);
         
         // Optionally log or send alert
         if(UseAISuggestions && StringLen(aiSuggestions) > 0)
         {
            PrintFormat("[%s] AI Suggestions: %s", stratNames[s], aiSuggestions);
         }
         
         continue; // Skip this trade
      }
      
      PrintFormat("[%s] AI APPROVED trade - Confidence %.2f", stratNames[s], aiConfidence);
      // ==============================================
      // END AI CHECK
      // ==============================================

      // --- SAFETY: compute current potential risk in USD across open EA trades
      double currentPotentialRiskUSD = 0.0;
      int openEAOrders = 0;
      for(int oi = 0; oi < OrdersTotal(); oi++)
      {
         if(!OrderSelect(oi, SELECT_BY_POS, MODE_TRADES)) continue;
         if(OrderSymbol() != TradeSymbol) continue;
         if(OrderMagicNumber() != MagicNumber) continue;
         if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;

         double oopen = OrderOpenPrice();
         double osl = OrderStopLoss();
         if(osl <= 0) { openEAOrders++; continue; } // skip open trades without SL from risk calc
         double odistance = MathAbs(oopen - osl);
         double opoint = MarketInfo(TradeSymbol, MODE_POINT); if(opoint <= 0) opoint = 0.01;
         double ostoppips = odistance / opoint;
         double otickvalue = MarketInfo(TradeSymbol, MODE_TICKVALUE);
         if(otickvalue <= 0) otickvalue = opoint * 10.0;
         double lossPerLotExisting = ostoppips * otickvalue;
         currentPotentialRiskUSD += lossPerLotExisting * OrderLots();
         openEAOrders++;
      }

      // total allowed risk budget (dollars)
      double totalRiskBudgetUSD = AccountEquity() * (TotalDailyRiskPct / 100.0);
      if(totalRiskBudgetUSD <= 0) totalRiskBudgetUSD = AccountEquity() * 0.08; // fallback

      // quick concurrent trade check
      if(openEAOrders >= MaxConcurrentTrades) {
         PrintFormat("[%s] MaxConcurrentTrades reached (%d) - skip entry.", stratNames[s], MaxConcurrentTrades);
         continue;
      }

      // compute allowed USD risk for new trade
      double allowedNewTradeRiskUSD = totalRiskBudgetUSD - currentPotentialRiskUSD;
      if(allowedNewTradeRiskUSD <= 0) {
         PrintFormat("[%s] No remaining risk budget (%.2f USD) - skip entry.", stratNames[s], allowedNewTradeRiskUSD);
         continue;
      }

      // Determine stopDistance using ATR on EntryTF
      double stopDistance = ComputeSLPriceFromATR(sig, EntryTF);
      if(stopDistance <= 0)
      {
         PrintFormat("[%s] Computed stopDistance invalid (%.5f), skipping.", stratNames[s], stopDistance);
         continue;
      }

      // Use current market price for calculations (not closed bar price)
      RefreshRates();
      double entryPrice = (sig == 1) ? MarketInfo(TradeSymbol, MODE_ASK) : MarketInfo(TradeSymbol, MODE_BID);
      double sl = 0, tp = 0;
      
      if(sig == 1)
      {
         sl = entryPrice - stopDistance;
         tp = entryPrice + stopDistance * RR_Ratio;
      }
      else
      {
         sl = entryPrice + stopDistance;
         tp = entryPrice - stopDistance * RR_Ratio;
      }

      int digits = (int)MarketInfo(TradeSymbol, MODE_DIGITS);
      sl = NormalizeDouble(sl, digits);
      tp = NormalizeDouble(tp, digits);

      if(sl <= 0 || tp <= 0) {
         PrintFormat("[%s] Invalid SL (%.5f) or TP (%.5f), skipping.", stratNames[s], sl, tp);
         continue;
      }

      // Lot sizing for this signal - safer approach using dollar risk budget
      double lots = LotsForDollarRisk(stopDistance, allowedNewTradeRiskUSD);
      if(lots <= 0) { 
         PrintFormat("[%s] LotsForDollarRisk returned <=0 (%.2f), skipping.", stratNames[s], lots); 
         continue; 
      }

      // Also ensure per-trade percent cap is respected (optional safeguard)
      double perTradeRiskUSD_cap = AccountEquity() * (RiskPercent / 100.0);
      double lots_from_percent = LotsForDollarRisk(stopDistance, perTradeRiskUSD_cap);
      if(lots > lots_from_percent) {
         lots = lots_from_percent;
      }

      // Apply MaxLossPerTradeUSD cap
      double lots_by_losscap = LotsForDollarRisk(stopDistance, MaxLossPerTradeUSD);
      if(lots > lots_by_losscap) {
         PrintFormat("[%s] Lot reduced by MaxLossPerTradeUSD cap: %.2f -> %.2f", stratNames[s], lots, lots_by_losscap);
         lots = lots_by_losscap;
      }

      // Ensure not to exceed remainingLotsAllowed
      if(lots > remainingLotsAllowed) {
         lots = remainingLotsAllowed;
         PrintFormat("[%s] Lot size limited by remaining allowance: %.2f", stratNames[s], lots);
      }
      
      lots = NormalizeLotSteps(lots);
      if(lots < minLot)
      {
         PrintFormat("[%s] Final lot size %.2f < minlot %.2f, skipping.", stratNames[s], lots, minLot);
         continue;
      }

      // Final broker limit check
      double brokerMax = MarketInfo(TradeSymbol, MODE_MAXLOT);
      if(brokerMax <= 0) brokerMax = maxlot;
      double effectiveMax = MathMin(brokerMax, maxlot);
      if(lots > effectiveMax) lots = effectiveMax;

      // Send order
      int ticket = trade_SendOrder(MagicNumber, sig, lots, sl, tp);
      if(ticket > 0)
      {
         remainingLotsAllowed -= lots;
         PrintFormat("[%s] Opened %s ticket=%d lots=%.2f (remaining allowance %.2f)", 
                     stratNames[s], SideName(sig), ticket, lots, remainingLotsAllowed);
      }
      else
      {
         PrintFormat("[%s] OrderSend failed for side %s.", stratNames[s], SideName(sig));
      }

      // If no remaining allotment, break
      if(remainingLotsAllowed < minLot) break;
   }
}

// INITIALIZATION & CLEANUP
void OnInit()
{
   lastEntryBarTime = 0;
   lastTradeBarTime = 0;
   dayStartBalance = 0.0;
   dayStartDate = -1;
   dailyCapHit = false;
   positionCount = 0;
   
   for(int i = 0; i < 100; i++) {
      openPositions[i].ticket = -1;
      openPositions[i].partialClosed = false;
      openPositions[i].movedToBE = false;
   }
   
   if(RiskPercent <= 0 || RiskPercent > 10) {
      PrintFormat("WARNING: Risk percent %.2f may be too high or invalid", RiskPercent);
   }
   
   if(ATRMultiplier <= 0) {
      Print("ERROR: ATR Multiplier must be positive");
      return;
   }
   
   if(RR_Ratio <= 0) {
      Print("ERROR: Risk-Reward Ratio must be positive");
      return;
   }
   
   if(!UseTrendFollowing && !UseMeanReversion && !UseBreakout) {
      Print("WARNING: No trading strategies enabled");
   }
   
   if(UseAIConfidence)
   {
      PrintFormat("AI Confidence enabled. Server: %s MinConfidence: %.2f", 
                  AIServerURL, MinAIConfidence);
      
      // Test connection to AI server
      string testResponse = SendHTTPRequest("http://127.0.0.1:5000/health", "{}");
      if(StringLen(testResponse) > 0)
      {
         Print("AI Server connection test: SUCCESS");
         PrintFormat("Response: %s", testResponse);
      }
      else
      {
         Print("AI Server connection test: FAILED - EA will allow trades when server unreachable");
      }
   }

   // Market info sanity check (important for XAU)
   double tv = MarketInfo(TradeSymbol, MODE_TICKVALUE);
   double pt = MarketInfo(TradeSymbol, MODE_POINT);
   double mr = MarketInfo(TradeSymbol, MODE_MARGINREQUIRED);
   PrintFormat("MarketInfo sanity: TICKVALUE=%.6f POINT=%.6f MARGINREQUIRED=%.2f", tv, pt, mr);
   if(tv <= 0 || pt <= 0) {
      Print("WARNING: tickvalue/point appear invalid. Recheck broker symbol specification. Using fallbacks is risky.");
   }

   PrintFormat("GoldBot EA initialized. Symbol=%s TF=%s Risk=%.2f%% MaxLot=%.2f", 
               TradeSymbol, EnumToString(EntryTF), RiskPercent, maxlot);
   PrintFormat("Strategies: Trend=%s MeanRev=%s Breakout=%s CloseOpposites=%s CloseAllOnDailyCap=%s", 
               UseTrendFollowing ? "ON" : "OFF",
               UseMeanReversion ? "ON" : "OFF", 
               UseBreakout ? "ON" : "OFF",
               CloseOpposites ? "ON" : "OFF",
               CloseAllOnDailyCap ? "ON" : "OFF");
}

void OnDeinit(const int reason)
{
   PrintFormat("GoldBoT EA deinitialized. Reason: %d", reason);
}