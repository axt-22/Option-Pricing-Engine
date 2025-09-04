from flask import Flask, render_template, request, jsonify
import yfinance as yf
from datetime import date, datetime, timedelta
from pricing_models import Option, MarketData, BlackScholes, Binomial, MonteCarlo

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_option_data', methods=['POST'])
def get_option_data():
    ticker_symbol = request.form['ticker']
    ticker = yf.Ticker(ticker_symbol)
    expiries = ticker.options
    chosen_expiry = expiries[2] if len(expiries) >= 3 else expiries[0]

    option_chain = ticker.option_chain(chosen_expiry)
    calls = option_chain.calls
    puts = option_chain.puts

    strike_prices = sorted(list(set(calls['strike']).union(set(puts['strike']))))

    return jsonify({'expiries': expiries, 'strikes': strike_prices})

@app.route('/price_option', methods=['POST'])
def price_option():
    data = request.form
    ticker_symbol = data['ticker']
    option_type = data['option_type']
    style = data['style']
    model_choice = data['model']
    strike = float(data['strike'])
    expiry = data['expiry']

    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.info['regularMarketPrice']
    option_chain = ticker.option_chain(expiry)
    option_df = option_chain.calls if option_type == 'call' else option_chain.puts

    # Select the row matching the chosen strike
    match = option_df[option_df['strike'] == strike]
    if match.empty:
        return jsonify({'error': 'Strike price not available in option chain'}), 400

    iv = float(match['impliedVolatility'].values[0])
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - date.today()).days

    option = Option(option_type, style, strike, ticker_symbol, days_to_expiry)
    marketdata = MarketData(spot_price, iv, 0.05, 0)

    if model_choice == 'Black-Scholes':
        model = BlackScholes(option, marketdata)
        price = round(model.price(), 2)
        delta, gamma, vega, theta, rho = model.greeks()
    elif model_choice == 'Monte Carlo':
        model = MonteCarlo(option, marketdata)
        price = round(model.price(sim=10000, time_step=50), 2)
        delta, gamma, vega, theta, rho = model.greeks()
    elif model_choice == 'Binomial Tree':
        model = Binomial(option, marketdata)
        option_tree, price_tree = model.price(N=500)
        price = round(option_tree[0][0], 2)
        delta, gamma, vega, theta, rho = model.greeks(option_tree, price_tree)
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    return jsonify({
        'price': price,
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'vega': round(vega, 4),
        'theta': round(theta, 4),
        'rho': round(rho, 4),
        'spot_price': round(spot_price, 2)
        })

if __name__ == '__main__':
    app.run(debug=True)
