#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: AxiaCore S.A.S. http://axiacore.com
#
# Based on js-expression-eval, by Matthew Crumley (email@matthewcrumley.com, http://silentmatt.com/)
# https://github.com/silentmatt/js-expression-eval
#
# Ported to Python and modified by Vera Mazhuga (ctrl-alt-delete@live.com, http://vero4ka.info/)
#
# You are free to use and modify this code in anyway you find useful. Please leave this comment in the code
# to acknowledge its original source. If you feel like it, I enjoy hearing about projects that use my code,
# but don't feel like you have to let me know or ask permission.
from __future__ import division

import math
import random
import re
import numpy as np
# import Dataview
import pandas as pd
from sklearn.linear_model import LinearRegression
import common.datetime_process as dtp
import copy
TNUMBER = 0
TOP1 = 1
TOP2 = 2
TVAR = 3
TFUNCALL = 4


class Token():

    def __init__(self, type_, index_, prio_, number_):
        self.type_ = type_
        self.index_ = index_ or 0
        self.prio_ = prio_ or 0
        self.number_ = number_ if number_ != None else 0

    def toString(self):
        if self.type_ == TNUMBER:
            return self.number_
        if self.type_ == TOP1 or self.type_ == TOP2 or self.type_ == TVAR:
            return self.index_
        elif self.type_ == TFUNCALL:
            return 'CALL'
        else:
            return 'Invalid Token'


class Expression():

    def __init__(self, tokens, ops1, ops2, functions):
        self.tokens = tokens
        self.ops1 = ops1
        self.ops2 = ops2
        self.functions = functions

    def simplify(self, values):
        values = values or {}
        nstack = []
        newexpression = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TNUMBER:
                nstack.append(item)
            elif type_ == TVAR and item.index_ in values:
                item = Token(TNUMBER, 0, 0, values[item.index_])
                nstack.append(item)
            elif type_ == TOP2 and len(nstack) > 1:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = self.ops2[item.index_]
                item = Token(TNUMBER, 0, 0, f(n1.number_, n2.number_))
                nstack.append(item)
            elif type_ == TOP1 and nstack:
                n1 = nstack.pop()
                f = self.ops1[item.index_]
                item = Token(TNUMBER, 0, 0, f(n1.number_))
                nstack.append(item)
            else:
                while len(nstack) > 0:
                    newexpression.append(nstack.pop(0))
                newexpression.append(item)
        while nstack:
            newexpression.append(nstack.pop(0))

        return Expression(newexpression, self.ops1, self.ops2, self.functions)

    def substitute(self, variable, expr):
        if not isinstance(expr, Expression):
            expr = Parser().parse(str(expr))
        newexpression = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TVAR and item.index_ == variable:
                for j in range(0, len(expr.tokens)):
                    expritem = expr.tokens[j]
                    replitem = Token(
                        expritem.type_,
                        expritem.index_,
                        expritem.prio_,
                        expritem.number_,
                    )
                    newexpression.append(replitem)
            else:
                newexpression.append(item)

        ret = Expression(newexpression, self.ops1, self.ops2, self.functions)
        return ret

    def evaluate(self, values):
        values = values or {}
        nstack = []
        L = len(self.tokens)
        for item in self.tokens:
            type_ = item.type_
            if type_ == TNUMBER:
                nstack.append(item.number_)
            elif type_ == TOP2:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = self.ops2[item.index_]
                nstack.append(f(n1, n2))
            elif type_ == TVAR:
                if item.index_ in values:
                    nstack.append(values[item.index_])
                elif item.index_ in self.functions:
                    nstack.append(self.functions[item.index_])
                else:
                    raise Exception('undefined variable: ' + item.index_)
            elif type_ == TOP1:
                n1 = nstack.pop()
                f = self.ops1[item.index_]
                nstack.append(f(n1))
            elif type_ == TFUNCALL:
                n1 = nstack.pop()
                f = nstack.pop()
                if callable(f):
                    if type(n1) is list:
                        nstack.append(f(*n1))
                    else:
                        nstack.append(f(n1))
                else:
                    raise Exception(f + ' is not a function')
            else:
                raise Exception('invalid Expression')
        if len(nstack) > 1:
            raise Exception('invalid Expression (parity)')
        return nstack[0]

    def toString(self, toJS=False):
        nstack = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TNUMBER:
                if type(item.number_) == str:
                    nstack.append("'" + item.number_ + "'")
                else:
                    nstack.append(item.number_)
            elif type_ == TOP2:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = item.index_
                if toJS and f == '^':
                    nstack.append('math.pow(' + n1 + ',' + n2 + ')')
                else:
                    frm = '({n1}{f}{n2})'
                    if f == ',':
                        frm = '{n1}{f}{n2}'

                    nstack.append(frm.format(
                        n1=n1,
                        n2=n2,
                        f=f,
                    ))


            elif type_ == TVAR:
                nstack.append(item.index_)
            elif type_ == TOP1:
                n1 = nstack.pop()
                f = item.index_
                if f == '-':
                    nstack.append('(' + f + str(n1) + ')')
                else:
                    nstack.append(f + '(' + str(n1) + ')')
            elif type_ == TFUNCALL:
                n1 = nstack.pop()
                f = nstack.pop()
                nstack.append(f + '(' + n1 + ')')
            else:
                raise Exception('invalid Expression')
        if len(nstack) > 1:
            raise Exception('invalid Expression (parity)')
        return nstack[0]

    def __str__(self):
        return self.toString()

    def symbols(self):
        vars = []
        for i in range(0, len(self.tokens)):
            item = self.tokens[i]
            if item.type_ == TVAR and not item.index_ in vars:
                vars.append(item.index_)
        return vars

    def variables(self):
        return [
            sym for sym in self.symbols()
            if sym not in self.functions]


class Parser:
    class Expression(Expression):
        pass

    PRIMARY = 1
    OPERATOR = 2
    FUNCTION = 4
    LPAREN = 8
    RPAREN = 16
    COMMA = 32
    SIGN = 64
    CALL = 128
    NULLARY_CALL = 256

    @staticmethod
    def to_weights(group, demeaned):
        if demeaned:
            demeaned_vals = group - group.mean()
            if (len(demeaned_vals) != 1) or (any(demeaned_vals.tolist())):
                return demeaned_vals / demeaned_vals.abs().sum()
            else:
                demeaned_vals.values[0] = 0.0
                return demeaned_vals
        else:
            return group / group.abs().sum()

    def mkt_(self,returns,cap):
        cap_w = cap.groupby('date').apply(Parser.to_weights,False)
        # print((returns.multiply(cap_w)).groupby('date').sum())
        return (returns.multiply(cap_w)).groupby('date').sum()
    """
    求平均回报率 = 流通市值加权的所有股票回报率的总和？？
    """

    def smb_(self,returns,cap,cir_cap):
        def quantile_calc(x,q):
            return pd.qcut(x, q=q, labels=False)
        # dateindex = cap.index.levels[0]
        # assetindex = cap.index.levels[1]
        pf = cap.groupby('date').apply(quantile_calc,3)
        # print(returns)
        # print(pf)

        upper_stk = pf[pf==2].index.tolist()
        lower_stk = pf[pf==0].index.tolist()
        up_return = self.mkt_(returns.loc[upper_stk],cir_cap.loc[upper_stk])
        low_return = self.mkt_(returns.loc[lower_stk],cir_cap.loc[lower_stk])
        print(up_return-low_return)
        return up_return-low_return

    def hml_(self,returns,pb,cir_cap):

        def quantile_calc(x,q):
            return pd.qcut(x, q=q, labels=False)
        pf = pb.groupby('date').apply(quantile_calc,3)
        upper_stk = pf[pf==2].index.tolist()
        lower_stk = pf[pf==0].index.tolist()
        up_return = self.mkt_(returns.loc[upper_stk],cir_cap.loc[upper_stk])
        low_return = self.mkt_(returns.loc[lower_stk],cir_cap.loc[lower_stk])
        print(up_return-low_return)
        return up_return-low_return

    def ivol(self,r_,mkt,smb,hml,delta = 30):
        ivol = pd.DataFrame(index=r_.index)
        r = pd.DataFrame(r_).dropna()
        dateindex = r.index.levels[0]
        assetindex = r.index.levels[1]
        for date in dateindex:
            end_date = date
            start_date = dtp.shift_time_(date,-delta)
            datelist = mkt[(mkt.index<=end_date)&(mkt.index>=start_date)].index

            mkt_ = mkt[datelist]
            smb_ = smb[datelist]
            hml_ = hml[datelist]
            # print(smb_)
            if(len(mkt_)>=10 and len(smb_)>10 and len(mkt_)==len(smb_)):
                x_train = pd.concat([mkt_,smb_,hml_],axis=1)
                for asset in r.loc[date].index:
                    # pointer = 'date>="{}" and date<="{}" and asset == "{}" '.format(start_date, end_date,asset)
                    # y_train = r.query(pointer)
                    y_train = r.loc[(datelist,asset),:]
                    if (len(y_train)==len(mkt_)):
                        linreg = LinearRegression()
                        linreg.fit(x_train,y_train)
                        e_ivol = np.std((linreg.predict(X=x_train)-y_train),
                                        ddof = 1)
                        ivol.loc[(date,asset),'ivol'] = e_ivol[0]
        # print('==================================================')
        # print(ivol)
        return ivol

    def trade_when(self,tradesign,alpha,exitsign):

        # trade_index = tradesign.index
        # exit_index = exitsign.index
        # print(trade_index.intersection(exit_index))
        alpha_index = tradesign.index.intersection(exitsign.index)
        alpha_when = pd.DataFrame(index=alpha_index,columns=['trade_when'])
        # print(alpha_when)
        pre_date = alpha_index.levels[0][0]
        print(alpha_index.levels[0])
        last_date = pre_date
        print('Pre date inited...',pre_date)
        ticker = 0
        for pix in alpha_index:
                date,asset = pix

                if exitsign[pix]>0:
                    # print('Triggering Exits.')
                    alpha_when.loc[pix] = np.nan
                elif tradesign.loc[pix]>0:
                    # print('Triggering Alphas.')
                    alpha_when.loc[pix] = alpha.loc[pix]
                else:
                    # ticker = ticker+1
                    # print(ticker)
                    if (pre_date,asset) in alpha_index:
                        print('Triggering back valid alphas.')
                        # print(exitsign[pix],tradesign[pix])
                        print(pre_date,date)
                        alpha_when.loc[pix] = alpha_when.loc[(pre_date,asset)]

                    else:
                        print('Triggering back NaN.')
                        alpha_when.loc[pix] = np.nan

                if date != last_date:
                    pre_date = last_date
                    # print(last_date,date)
                last_date = date
                # if (alpha_index.levels[0][i-1],asset) not in
                # print (tradesign.loc[i])
            # elif exitsign>0:
            #     print('HAVEN\'T FINISHED YET.')
        # print()
        return alpha_when.sort_index()

    def zscore(self,x):
        if x.std(0) !=0:
            return (x-x.mean(0))/(x.std(0))
        else:
            return (x-x.mean(0))

    def group_zscore(self,x,group):
        def to_weights(df):
            demeaned_vals = (df - df.mean())
            if (len(demeaned_vals) != 1) or (any(demeaned_vals.tolist())):
                demeaned_vals = demeaned_vals / df.std()
            else:
                demeaned_vals.values[0] = 0.0
            return demeaned_vals
        dt = pd.concat([x,group],axis=1)
        gzscore = dt.groupby(['date','group'])[x.name].apply(to_weights)
        return gzscore

    def kth_element(self,x,d,k=1):
        df = pd.Series(index=x.index)
        date_ = x.index.levels[0]
        for i,p in enumerate(date_):
            pt = x.loc[((date_[max(0,i-d+1):i+1],slice(None)))].sort_values()
            if len(pt.index)!=0:
                df.loc[((date_[i],slice(None)))] = x.loc[((date_[max(0, i - d + 1):i + 1], slice(None)))].sort_values().iloc[-k]
            # kth = [x.loc[(date_[max(0,i-d+1):i+1],slice(None))] for i,p in enumerate(x.index)]
        return df

    def group_neutralize(self,a,group):
        grouper = [a.index.get_level_values('date','group')]
        return a.groupby

    def ts_delta(self,a,n):
        delta_ =  a - a.groupby(level='asset').shift(n)
        return delta_

    def rank(self,a):
        def get_rank(x):
            return ((x.rank()-1)/len(x))
        rank_ = a.groupby(level = 'date').apply(get_rank)
        return rank_

    def cbrt(self,a):
        return a.apply(np.cbrt)

    def stdscaler(self,a):
        from sklearn.preprocessing import StandardScaler
        a = a.to_numpy().reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(a)
        return scaler.transform(a)

    def purify(self,a):
        return a.replace([np.inf,-np.inf],np.nan)

    def and_logic(self,a,b):
        return a & b

    def log(self,a):
        return a.apply(np.log)


    def frac(self,a):
        return a - np.floor(a)

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a / b

    def mod(self, a, b):
        return a % b

    def sqrt(self,a):
        return a.apply(np.sqrt)

    def square(self,a):
        return a.apply(np.square)

    def concat(self, a, b, *args):
        result = u'{0}{1}'.format(a, b)
        for arg in args:
            result = u'{0}{1}'.format(result, arg)
        return result

    def equal(self, a, b):
        return a == b

    def notEqual(self, a, b):
        return a != b

    def greaterThan(self, a, b):
        return a > b

    def lessThan(self, a, b):
        return a < b

    def greaterThanEqual(self, a, b):
        return a >= b

    def lessThanEqual(self, a, b):
        return a <= b

    def andOperator(self, a, b):
        return (a and b)

    def orOperator(self, a, b):
        return (a or b)

    def neg(self, a):
        return -a

    def sigmoid(self,a):
        def sigmoid_fun(x):
            return 1 / (1 + np.exp(-x))
        return a.apply(sigmoid_fun)

    def random(self, a):
        return math.random() * (a or 1)

    def fac(self, a):  # a!
        return math.factorial(a)

    def pyt(self, a, b):
        return math.sqrt(a * a + b * b)

    def roll(self, a, b):
        rolls = []
        roll = 0
        final = 0
        for c in range(1, a):
            roll = random.randint(1, b)
            rolls.append(roll)
        return rolls

    def ifFunction(self, a, b, c):
        return b if a else c

    def append(self, a, b):
        if type(a) != list:
            return [a, b]
        a.append(b)
        return a

    def zero(self,a):
        return (a * 0.0)

    def freq(self,a):
        # def __freq(x):
        #     return x.value_counts() / len(x)
        freq_counter = a.value_counts()
        x = copy.deepcopy(a)
        for i in x.index:
            x.loc[i] = freq_counter[a.loc[i]]
        return x / len(x)

    def __init__(self):
        self.success = False
        self.errormsg = ''
        self.expression = ''

        self.pos = 0

        self.tokennumber = 0
        self.tokenprio = 0
        self.tokenindex = 0
        self.tmpprio = 0

        self.ops1 = {
            'square':self.square,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sqrt': self.sqrt,
            'abs': abs,
            'log':self.log,
            'ceil': np.ceil,
            'floor': np.floor,
            'round': round,
            '-': self.neg,
            'exp': np.exp,
            'frac':self.frac,
            'purify':self.purify,
            'rank':self.rank,
            'groupby':self.group_neutralize,
            'zscore':self.zscore,
            'sigmoid':self.sigmoid,
            'cbrt':self.cbrt,
            'stdscaler':self.stdscaler,
            'zero':self.zero,
            'freq':self.freq,

        }

        self.ops2 = {
            '+': self.add,
            '-': self.sub,
            '*': self.mul,
            '/': self.div,
            '%': self.mod,
            '^': math.pow,
            '**': math.pow,
            ',': self.append,
            '||': self.concat,
            "==": self.equal,
            "!=": self.notEqual,
            ">": self.greaterThan,
            "<": self.lessThan,
            ">=": self.greaterThanEqual,
            "<=": self.lessThanEqual,
            "and": self.andOperator,
            "or": self.orOperator,
            "&": self.and_logic,
            "D": self.roll,


        }

        self.functions = {
            'ts_delta':self.ts_delta,
            'random': random,
            'fac': self.fac,
            'log': math.log,
            'min': np.min,
            'max': np.max,
            'pyt': self.pyt,
            'pow': np.power,
            'atan2': math.atan2,
            'concat': self.concat,
            'if': self.ifFunction,
            'mkt': self.mkt_,
            'smb':self.smb_,
            'hml':self.hml_,
            'ivol':self.ivol,
            'trade_when': self.trade_when,
            'group_zscore':self.group_zscore,
            'kth_element': self.kth_element,
        }

        self.consts = {
            'E': math.e,
            'PI': math.pi,
        }

        self.values = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sqrt': math.sqrt,
            'log': math.log,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'round': round,
            'random': self.random,
            'fac': self.fac,
            'exp': math.exp,
            'min': min,
            'max': max,
            'pyt': self.pyt,
            'pow': math.pow,
            'atan2': math.atan2,
            'E': math.e,
            'PI': math.pi
        }

    def parse(self, expr):
        self.errormsg = ''
        self.success = True
        operstack = []
        tokenstack = []
        self.tmpprio = 0
        expected = self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
        noperators = 0
        self.expression = expr
        self.pos = 0

        while self.pos < len(self.expression):
            if self.isOperator():
                if self.isSign() and expected & self.SIGN:
                    if self.isNegativeSign():
                        self.tokenprio = 5
                        self.tokenindex = '-'
                        noperators += 1
                        self.addfunc(tokenstack, operstack, TOP1)
                    expected = \
                        self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
                elif self.isComment():
                    pass
                else:
                    if expected and self.OPERATOR == 0:
                        self.error_parsing(self.pos, 'unexpected operator')
                    noperators += 2
                    self.addfunc(tokenstack, operstack, TOP2)
                    expected = \
                        self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
            elif self.isNumber():
                if expected and self.PRIMARY == 0:
                    self.error_parsing(self.pos, 'unexpected number')
                token = Token(TNUMBER, 0, 0, self.tokennumber)
                tokenstack.append(token)
                expected = self.OPERATOR | self.RPAREN | self.COMMA
            elif self.isString():
                if (expected & self.PRIMARY) == 0:
                    self.error_parsing(self.pos, 'unexpected string')
                token = Token(TNUMBER, 0, 0, self.tokennumber)
                tokenstack.append(token)
                expected = self.OPERATOR | self.RPAREN | self.COMMA
            elif self.isLeftParenth():
                if (expected & self.LPAREN) == 0:
                    self.error_parsing(self.pos, 'unexpected \"(\"')
                if expected & self.CALL:
                    noperators += 2
                    self.tokenprio = -2
                    self.tokenindex = -1
                    self.addfunc(tokenstack, operstack, TFUNCALL)
                expected = \
                    self.PRIMARY | self.LPAREN | self.FUNCTION | \
                    self.SIGN | self.NULLARY_CALL
            elif self.isRightParenth():
                if expected & self.NULLARY_CALL:
                    token = Token(TNUMBER, 0, 0, [])
                    tokenstack.append(token)
                elif (expected & self.RPAREN) == 0:
                    self.error_parsing(self.pos, 'unexpected \")\"')
                expected = \
                    self.OPERATOR | self.RPAREN | self.COMMA | \
                    self.LPAREN | self.CALL
            elif self.isComma():
                if (expected & self.COMMA) == 0:
                    self.error_parsing(self.pos, 'unexpected \",\"')
                self.addfunc(tokenstack, operstack, TOP2)
                noperators += 2
                expected = \
                    self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
            elif self.isConst():
                if (expected & self.PRIMARY) == 0:
                    self.error_parsing(self.pos, 'unexpected constant')
                consttoken = Token(TNUMBER, 0, 0, self.tokennumber)
                tokenstack.append(consttoken)
                expected = self.OPERATOR | self.RPAREN | self.COMMA
            elif self.isOp2():
                if (expected & self.FUNCTION) == 0:
                    self.error_parsing(self.pos, 'unexpected function')
                self.addfunc(tokenstack, operstack, TOP2)
                noperators += 2
                expected = self.LPAREN
            elif self.isOp1():
                if (expected & self.FUNCTION) == 0:
                    self.error_parsing(self.pos, 'unexpected function')
                self.addfunc(tokenstack, operstack, TOP1)
                noperators += 1
                expected = self.LPAREN
            elif self.isVar():
                if (expected & self.PRIMARY) == 0:
                    self.error_parsing(self.pos, 'unexpected variable')
                vartoken = Token(TVAR, self.tokenindex, 0, 0)
                tokenstack.append(vartoken)
                expected = \
                    self.OPERATOR | self.RPAREN | \
                    self.COMMA | self.LPAREN | self.CALL
            elif self.isWhite():
                pass
            else:
                if self.errormsg == '':
                    self.error_parsing(self.pos, 'unknown character')
                else:
                    self.error_parsing(self.pos, self.errormsg)
        if self.tmpprio < 0 or self.tmpprio >= 10:
            self.error_parsing(self.pos, 'unmatched \"()\"')
        while len(operstack) > 0:
            tmp = operstack.pop()
            tokenstack.append(tmp)
        if (noperators + 1) != len(tokenstack):
            self.error_parsing(self.pos, 'parity')

        return Expression(tokenstack, self.ops1, self.ops2, self.functions)

    def evaluate(self, expr, variables):
        return self.parse(expr).evaluate(variables)

    def error_parsing(self, column, msg):
        self.success = False
        self.errormsg = 'parse error [column ' + str(column) + ']: ' + msg
        raise Exception(self.errormsg)

    def addfunc(self, tokenstack, operstack, type_):
        operator = Token(
            type_,
            self.tokenindex,
            self.tokenprio + self.tmpprio,
            0,
        )
        while len(operstack) > 0:
            if operator.prio_ <= operstack[len(operstack) - 1].prio_:
                tokenstack.append(operstack.pop())
            else:
                break
        operstack.append(operator)

    def isNumber(self):
        r = False

        if self.expression[self.pos] == 'E':
            return False

        # number in scientific notation
        pattern = r'([-+]?([0-9]*\.?[0-9]*)[eE][-+]?[0-9]+).*'
        match = re.match(pattern, self.expression[self.pos:])
        if match:
            self.pos += len(match.group(1))
            self.tokennumber = float(match.group(1))
            return True

        # number in decimal
        str = ''
        while self.pos < len(self.expression):
            code = self.expression[self.pos]
            if (code >= '0' and code <= '9') or code == '.':
                if (len(str) == 0 and code == '.'):
                    str = '0'
                str += code
                self.pos += 1
                try:
                    self.tokennumber = int(str)
                except ValueError:
                    self.tokennumber = float(str)
                r = True
            else:
                break
        return r

    def unescape(self, v, pos):
        buffer = []
        escaping = False

        for i in range(0, len(v)):
            c = v[i]

            if escaping:
                if c == "'":
                    buffer.append("'")
                    break
                elif c == '\\':
                    buffer.append('\\')
                    break
                elif c == '/':
                    buffer.append('/')
                    break
                elif c == 'b':
                    buffer.append('\b')
                    break
                elif c == 'f':
                    buffer.append('\f')
                    break
                elif c == 'n':
                    buffer.append('\n')
                    break
                elif c == 'r':
                    buffer.append('\r')
                    break
                elif c == 't':
                    buffer.append('\t')
                    break
                elif c == 'u':
                    # interpret the following 4 characters
                    # as the hex of the unicode code point
                    codePoint = int(v[i + 1, i + 5], 16)
                    buffer.append(unichr(codePoint))
                    i += 4
                    break
                else:
                    raise self.error_parsing(
                        pos + i,
                        'Illegal escape sequence: \'\\' + c + '\'',
                    )
                escaping = False
            else:
                if c == '\\':
                    escaping = True
                else:
                    buffer.append(c)

        return ''.join(buffer)

    def isString(self):
        r = False
        str = ''
        startpos = self.pos
        if self.pos < len(self.expression) and self.expression[self.pos] in ("'", "\""):
            quote_type = self.expression[self.pos]
            self.pos += 1
            while self.pos < len(self.expression):
                code = self.expression[self.pos]
                if code != quote_type or (str != '' and str[-1] == '\\'):
                    str += self.expression[self.pos]
                    self.pos += 1
                else:
                    self.pos += 1
                    self.tokennumber = self.unescape(str, startpos)
                    r = True
                    break
        return r

    def isConst(self):
        for i in self.consts:
            L = len(i)
            str = self.expression[self.pos:self.pos + L]
            if i == str:
                if len(self.expression) <= self.pos + L:
                    self.tokennumber = self.consts[i]
                    self.pos += L
                    return True
                if not self.expression[self.pos + L].isalnum() and self.expression[self.pos + L] != "_":
                    self.tokennumber = self.consts[i]
                    self.pos += L
                    return True
        return False

    def isOperator(self):
        ops = (
            ('+', 2, '+'),
            ('-', 2, '-'),
            ('**', 6, '**'),
            ('*', 3, '*'),
            (u'\u2219', 3, '*'),  # bullet operator
            (u'\u2022', 3, '*'),  # black small circle
            ('/', 4, '/'),
            ('%', 4, '%'),
            ('^', 6, '^'),
            ('||', 1, '||'),
            ('==', 1, '=='),
            ('!=', 1, '!='),
            ('<=', 1, '<='),
            ('>=', 1, '>='),
            ('<', 1, '<'),
            ('>', 1, '>'),
            ('and ', 0, 'and'),
            ('or ', 0, 'or'),
            ('&', 0, '&')
        )
        for token, priority, index in ops:
            if self.expression.startswith(token, self.pos):
                self.tokenprio = priority
                self.tokenindex = index
                self.pos += len(token)
                return True
        return False

    def isSign(self):
        code = self.expression[self.pos - 1]
        return (code == '+') or (code == '-')

    def isPositiveSign(self):
        code = self.expression[self.pos - 1]
        return code == '+'

    def isNegativeSign(self):
        code = self.expression[self.pos - 1]
        return code == '-'

    def isLeftParenth(self):
        code = self.expression[self.pos]
        if code == '(':
            self.pos += 1
            self.tmpprio += 10
            return True
        return False

    def isRightParenth(self):
        code = self.expression[self.pos]
        if code == ')':
            self.pos += 1
            self.tmpprio -= 10
            return True
        return False

    def isComma(self):
        code = self.expression[self.pos]
        if code == ',':
            self.pos += 1
            self.tokenprio = -1
            self.tokenindex = ","
            return True
        return False

    def isWhite(self):
        code = self.expression[self.pos]
        if code.isspace():
            self.pos += 1
            return True
        return False

    def isOp1(self):
        str = ''
        for i in range(self.pos, len(self.expression)):
            c = self.expression[i]
            if c.upper() == c.lower():
                if i == self.pos or (c != '_' and (c < '0' or c > '9')):
                    break
            str += c
        if len(str) > 0 and str in self.ops1:
            self.tokenindex = str
            self.tokenprio = 7
            self.pos += len(str)
            return True
        return False

    def isOp2(self):
        str = ''
        for i in range(self.pos, len(self.expression)):
            c = self.expression[i]
            if c.upper() == c.lower():
                if i == self.pos or (c != '_' and (c < '0' or c > '9')):
                    break
            str += c
        if len(str) > 0 and (str in self.ops2):
            self.tokenindex = str
            self.tokenprio = 7
            self.pos += len(str)
            return True
        return False

    def isVar(self):
        str = ''
        inQuotes = False
        for i in range(self.pos, len(self.expression)):
            c = self.expression[i]
            if c.lower() == c.upper():
                if ((i == self.pos and c != '"') or (not (c in '_."') and (c < '0' or c > '9'))) and not inQuotes:
                    break
            if c == '"':
                inQuotes = not inQuotes
            str += c
        if str:
            self.tokenindex = str
            self.tokenprio = 4
            self.pos += len(str)
            return True
        return False

    def isComment(self):
        code = self.expression[self.pos - 1]
        if code == '/' and self.expression[self.pos] == '*':
            self.pos = self.expression.index('*/', self.pos) + 2
            if self.pos == 1:
                self.pos = len(self.expression)
            return True
        return False
