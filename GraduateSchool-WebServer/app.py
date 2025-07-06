from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os

app = Flask(__name__)
app.secret_key = 'ChungbuckUniv'

BALANCE_FILE = 'balance_data.json'

# 사용자 정보 (패스워드만 메모리에 유지)
users = {
    'user1': {
        'password': generate_password_hash('pass123')
    },
    'user2': {
        'password': generate_password_hash('pass456')
    }
}

# 초기 잔액 생성 함수
def load_balance(userid):
    balances = {}
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, 'r') as f:
            balances = json.load(f)

    # 사용자 잔액이 없으면 기본 잔액으로 생성
    if userid not in balances:
        balances[userid] = 1000
        save_balance(balances)

    return balances[userid]


# 잔액 저장 함수
def save_balance(data):
    with open(BALANCE_FILE, 'w') as f:
        json.dump(data, f)

# 로그인 페이지
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']
        user = users.get(userid)

        if user and check_password_hash(user['password'], password):
            session['user'] = userid
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호를 확인해주세요.')
    return render_template('login.html')

# 로그아웃
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    userid = session['user']
    balances = {}
    user_balance = load_balance(userid)
    message = ""

    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, 'r') as f:
            balances = json.load(f)

    if request.method == 'POST':
        action = request.form['action']
        amount = int(request.form['amount'])

        if action == 'deposit':
            user_balance += amount
            message = f"{amount}원이 입금되었습니다."
        elif action == 'withdraw':
            if amount > user_balance:
                message = "잔액이 부족합니다."
            else:
                user_balance -= amount
                message = f"{amount}원이 출금되었습니다."

        # 변경된 잔액 저장
        balances[userid] = user_balance
        save_balance(balances)

    return render_template('dashboard.html', user=userid, balance=user_balance, message=message)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html"), 404

if __name__ == '__main__':
    app.run(debug=True)
