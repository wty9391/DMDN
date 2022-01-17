import sys
import os
import math
import collections
import codecs
import json

import numpy as np
from scipy.stats import norm
import pandas as pd

from util import Timer

epsilon = sys.float_info.epsilon

ipinyou_col = {
    "click": "click",
    "bid_price": "bidprice",
    "pay_price": "payprice",
    "feature_categorical": [
                "weekday",
                "hour",
                # "bidid",      # meaningless
                # "timestamp",  # meaningless
                # "logtype",    # meaningless
                # "ipinyouid",  # meaningless
                "useragent",  # special handling
                # "IP",         # dimension explosion
                "region",
                "city",
                "adexchange",
                "domain",
                # "url",         # dimension explosion
                "urlid",
                "slotwidth",
                "slotheight",
                "slotid",
                "slotvisibility",
                "slotformat",
                "creative",
                "keypage",
                "advertiser",
                "usertag"     # special handling
                            ],
    "feature_numerical" : ["slotprice"]
}

criteo_col = {
    "click": "click",
    "bid_price": None,
    "pay_price": "cost",
    "feature_categorical": [
        'campaign',
        'cat1',
        'cat2',
        'cat3',
        'cat4',
        'cat5',
        'cat6',
        'cat7',
        'cat8'
                            ],
    "feature_numerical" : []
}


class Dataset:
    def __init__(self):
        self


class Processor:
    os_list = ["windows",
               "ios",
               "mac",
               "android",
               "linux"]
    browser_list = [
                "chrome",
                "sogou",
                "maxthon",
                "safari",
                "firefox",
                "theworld",
                "opera",
                "ie"]
    def __init__(self, dataset, advertiser, root, logger):
        self.timer = Timer()
        self.dataset = dataset
        self.advertiser = advertiser
        self.root = root
        self.logger = logger
        self.train = None
        self.test = None
        self.usertag_list = set()
        self.usertag_dict = {}
        self.numerical_feature_dict = {}

    @staticmethod
    def useragent_process(content):
        """
        Process the useragent feature.
        :param content: useragent: str
        :return: 'os_browser': str
        """
        content = content.lower()
        os = "other"
        for o in Processor.os_list:
            if o in content:
                os = o
                break
        browser = "other"
        for b in Processor.browser_list:
            if b in content:
                browser = b
                break
        return os + "_" + browser

    @staticmethod
    def is_cache_available(output_dir):
        meta_path = os.path.join(output_dir, "metadata.json")
        return os.path.exists(meta_path)

    def process(self, output_dir):
        if self.dataset == "ipinyou":
            self.timer.tick()
            train_raw_data_path = os.path.join(self.root, self.advertiser, "train.log.txt")
            test_raw_data_path = os.path.join(self.root, self.advertiser, "test.log.txt")

            df_train = pd.read_csv(train_raw_data_path, sep="\t", header=0, index_col=False, keep_default_na=False, low_memory=False)
            df_test = pd.read_csv(test_raw_data_path, sep="\t", header=0, index_col=False, keep_default_na=False, low_memory=False)
            self.logger.info("Loading raw data finished, use time {:.2f}s".format(self.timer.tick()))

            self.usertag_list.clear()
            self.usertag_dict.clear()
            for usertags in df_train["usertag"].unique():
                self.usertag_list.update(usertags.split(","))
            self.usertag_list.add("other")
            for i, tag in enumerate(list(self.usertag_list)):
                self.usertag_dict[tag] = i

            self.numerical_feature_dict.clear()
            for feature in ipinyou_col["feature_numerical"]:
                self.numerical_feature_dict[feature] = {}
                self.numerical_feature_dict[feature]["mean"] = df_train[feature].mean()
                self.numerical_feature_dict[feature]["std"] = df_train[feature].std()

            self.logger.info("Scanning usertags/numerical features finished, use time {:.2f}s".format(self.timer.tick()))

            df_train = self._pre_process_ipinyou(df_train)
            df_test = self._pre_process_ipinyou(df_test)
            self.logger.info("pre-processing dataset finished, use time {:.2f}s".format(self.timer.tick()))
            df_train = self._process_ipinyou(df_train)
            df_test = self._process_ipinyou(df_test)
            self.logger.info("processing dataset finished, use time {:.2f}s".format(self.timer.tick()))

            self._cache_dataset(df_train, df_test, output_dir)

            self.train = df_train
            self.test = df_test

        elif self.dataset == "criteo":
            z_scale = 1e5
            train_test_split_day = 24

            self.timer.tick()
            raw_data_path = os.path.join(self.root, 'criteo_attribution_dataset.tsv.gz')
            df = pd.read_csv(raw_data_path, sep='\t', compression='gzip')
            self.logger.info("Loading raw data finished, use time {:.2f}s".format(self.timer.tick()))

            # no user tag, numerical feature to process
            self.usertag_list.clear()
            self.usertag_dict.clear()
            self.numerical_feature_dict.clear()

            df['day'] = np.floor(df.timestamp / 86400.).astype(int)
            df['cost'] = np.ceil(df['cost'] * z_scale)

            df_train = df[df.day < train_test_split_day]
            df_test = df[df.day >= train_test_split_day]

            # delete unused column
            df_train = self._pre_process_criteo(df_train)
            df_test = self._pre_process_criteo(df_test)
            self.logger.info("pre-processing dataset finished, use time {:.2f}s".format(self.timer.tick()))

            # clip z
            z_max = np.percentile(df['cost'].values.reshape((-1, 1)), 95)
            df_train.loc[:, 'cost'] = np.clip(df_train['cost'].values, 0, z_max)
            df_test.loc[:, 'cost'] = np.clip(df_test['cost'].values, 0, z_max)
            self.logger.info("processing dataset finished, use time {:.2f}s".format(self.timer.tick()))

            self._cache_dataset(df_train, df_test, output_dir)

            self.train = df_train
            self.test = df_test
        else:
            raise Exception("Unsupported dataset {}".format(self.dataset))

    def _pre_process_criteo(self, df):
        """
        1) remove unused columns
        :param df:
        :return:
        """
        used_columns = [criteo_col["click"], criteo_col["bid_price"], criteo_col["pay_price"]]
        used_columns.extend(criteo_col["feature_categorical"])
        used_columns.extend(criteo_col["feature_numerical"])

        all_columns = list(df.columns)
        unused_columns = [col for col in all_columns if col not in used_columns]
        df.drop(columns=unused_columns, inplace=True)
        return df

    def _pre_process_ipinyou(self, df):
        """
        1) remove unused columns

        :return:
        """

        used_columns = [ipinyou_col["click"], ipinyou_col["bid_price"], ipinyou_col["pay_price"]]
        used_columns.extend(ipinyou_col["feature_categorical"])
        used_columns.extend(ipinyou_col["feature_numerical"])

        all_columns = list(df.columns)
        unused_columns = [col for col in all_columns if col not in used_columns]
        df.drop(columns=unused_columns, inplace=True)
        return df


    def _process_ipinyou(self, df):
        """
        1) process user agent and user tag
        2) normalize numerical features

        :param df:
        :return:
        """

        # process user agent
        df["useragent"] = df.apply(lambda x: Processor.useragent_process(x["useragent"]), axis=1)

        # process user tag
        usertag_columns = ["usertag_{}".format(usertag) for usertag in list(self.usertag_list)]
        usertag_data = []
        other_index = self.usertag_dict["other"]

        for _, row in df.iterrows():
            usertags = [0] * len(self.usertag_list)
            for usertag in row["usertag"].split(","):
                usertags[self.usertag_dict.get(usertag, other_index)] = 1
            usertag_data.append(usertags)

        df_usertag = pd.DataFrame(columns=usertag_columns, data=usertag_data)

        assert len(df.index) == len(df_usertag.index)

        df_usertag.index = df.index
        df = pd.concat([df, df_usertag], axis=1)
        df.drop(columns=["usertag"], inplace=True)

        # normalize numerical features
        for feature, mean_variance in self.numerical_feature_dict.items():
            df[feature] = df.apply(lambda x: (x[feature] - mean_variance["mean"]) / (mean_variance["std"]+epsilon), axis=1)

        return df


    def _cache_dataset(self, train, test, root):
        meta_path = os.path.join(root, "metadata.json")
        train_path = os.path.join(root, "train.csv")
        test_path = os.path.join(root, "test.csv")

        train.to_csv(train_path, mode='w', index=False, sep='\t', header=True)
        test.to_csv(test_path, mode='w', index=False, sep='\t', header=True)

        obj = {"dataset": self.dataset,
               "advertiser": self.advertiser,
               "usertag_list": list(self.usertag_list),
               "usertag_dict": self.usertag_dict,
               "numerical_feature_dict": self.numerical_feature_dict}

        with codecs.open(meta_path, 'wb+', 'utf-8') as f:
            f.write(json.dumps(obj))


    def load_from_cache(self, root):
        meta_path = os.path.join(root, "metadata.json")

        if not os.path.exists(meta_path):
            error_message = "file not exists in {}".format(meta_path)
            self.logger.error(error_message)
            raise Exception(error_message)

        train_path = os.path.join(root, "train.csv")
        test_path = os.path.join(root, "test.csv")
        self.train = pd.read_csv(train_path, sep="\t", header=0, index_col=False, keep_default_na=False, low_memory=False)
        self.test = pd.read_csv(test_path, sep="\t", header=0, index_col=False, keep_default_na=False, low_memory=False)

        obj = json.load(codecs.open(meta_path, 'rb', 'utf-8'))
        self.dataset = obj["dataset"]
        self.advertiser = obj["advertiser"]
        self.usertag_list = set(obj["usertag_list"])
        self.usertag_dict = obj["usertag_dict"]
        self.numerical_feature_dict = obj["numerical_feature_dict"]

        return self


class Processor_market_price:
    def __init__(self, price_name='payprice', min_price=1, max_price=300, price_upper_bound=500):
        self.name_col = {}  # feature_name:origin_index
        self.pdf = {}  # price:probability
        self.cdf = {}  # price:probability
        self.number_record = 0
        self.price_name = price_name
        self.min_price = min_price
        self.max_price = max_price
        self.price_upper_bound = price_upper_bound
        self.data = []

        return

    def load(self, path):
        f = open(path, 'r', encoding="utf-8")
        first = True  # first line is header
        for l in f:
            s = l.split('\t')
            if first:
                for i in range(0, len(s)):
                    self.name_col[s[i].strip()] = i
                price_index = self.name_col[self.price_name]
                first = False
                continue
            price = float(s[price_index])+epsilon
            price_int = math.floor(price)
            price_int = price_int if price_int < self.price_upper_bound else self.price_upper_bound
            self.max_price = price_int if self.max_price < price_int else self.max_price

            if price_int in self.pdf:
                self.pdf[price_int] += 1
            else:
                self.pdf[price_int] = 1

            self.data.append(price_int)
            self.number_record += 1

        for price in range(self.min_price, self.max_price+1):
            if price not in self.pdf:
                self.pdf[price] = 0

        for price in self.pdf:
            self.pdf[price] = self.pdf[price]/self.number_record

        for price in self.pdf:
            p = 0
            for j in self.pdf:
                p += self.pdf[j] if j <= price else 0
            self.cdf[price] = p

        return self.cdf, self.pdf

    def load_by_array(self, z):
        # z is m x 1 matrix

        for s in z.tolist():
            price = float(s[0]) + epsilon
            price_int = math.floor(price)
            price_int = price_int if price_int < self.price_upper_bound else self.price_upper_bound
            self.max_price = price_int if self.max_price < price_int else self.max_price

            if price_int in self.pdf:
                self.pdf[price_int] += 1
            else:
                self.pdf[price_int] = 1

            self.data.append(price_int)
            self.number_record += 1

        for price in range(self.min_price, self.max_price+1):
            if price not in self.pdf:
                self.pdf[price] = 0

        for price in self.pdf:
            self.pdf[price] = self.pdf[price]/self.number_record

        for price in self.pdf:
            p = 0
            for j in self.pdf:
                p += self.pdf[j] if j <= price else 0
            self.cdf[price] = p

        return self.cdf, self.pdf

    def validate(self):
        if self.number_record == 0:
            print("This dataset processor has not been loaded")
            return False

        for i in self.pdf:
            p = 0
            for j in self.pdf:
                p += self.pdf[j] if j <= i else 0
            if self.cdf[i] != p:
                print("pdf: 0:{} is {}, cdf:{} is {}".format(str(i), str(p), str(i), str(self.cdf[i])))
                return False
        print("This dataset processor has validated successfully")
        return True

class Censored_processor_market_price:
    def __init__(self, min_price=1, max_price=300):
        self.min_price = min_price
        self.max_price = max_price
        self.truth = {"pdf": {}, "cdf": {}, "data": []}
        self.win = {"pdf": {}, "cdf": {}, "data": []}
        self.lose = {"pdf": {}, "cdf": {}, "data": []}
        self.bid = {"pdf": {}, "cdf": {}, "data": []}
        self.number_record = 0
        self.survive = {"pdf": {}, "cdf": {}, "data": {"b": [], "d": [], "n": []}}
        self.price_upper_bound = 500

        return

    def load(self, x, z, bidder):
        assert x.shape[0] == z.shape[0], "features' number must be equal prices' number"

        bids = bidder.bid(x).reshape((x.shape[0], 1))

        truncate = z > self.price_upper_bound
        z[truncate] = self.price_upper_bound

        win = bids > z
        lose = bids <= z

        self.max_price = max(z[:, 0])

        self.truth = self._count(z[:, 0].tolist())
        self.win = self._count(z[win].tolist())
        self.lose = self._count(z[lose].tolist())
        self.bid = self._count(bids[:, 0].tolist())
        self.number_record = x.shape[0]

        # fit survival model
        zs = list(range(self.min_price, self.max_price+1))

        counter_win_z = collections.Counter(z[win].tolist())
        counter_lose_bid = collections.Counter(bids[lose].tolist())

        counter_win_z_sum = {}
        counter_lose_bid_sum = {}

        for i in range(len(zs)):
            count_win = 0
            count_lose_bid = 0
            for b in zs[i:]:
                count_win = count_win + counter_win_z[b]
                count_lose_bid = count_lose_bid + counter_lose_bid[b]
            counter_win_z_sum[zs[i]] = count_win
            counter_lose_bid_sum[zs[i]] = count_lose_bid

        for b in zs:
            self.survive["data"]["b"].append(b)
            self.survive["data"]["d"].append(counter_win_z[b-1])
            self.survive["data"]["n"].append(counter_win_z_sum[b] + counter_lose_bid_sum[b])

        # calculate cdf
        for b in zs:
            pr_lose = 1.0
            for j in range(zs[0], b):
                index = self.survive["data"]["b"].index(j)
                if self.survive["data"]["n"][index] == 0:
                    pr_lose = 0
                else:
                    pr_lose = pr_lose * (self.survive["data"]["n"][index] - self.survive["data"]["d"][index])\
                        / self.survive["data"]["n"][index]
            self.survive["cdf"][b] = 1 - pr_lose if 1 - pr_lose <= 1.0 else self.survive["cdf"][b-1]
        self.survive["cdf"][0] = 1e-6  # in case of zero
        # calculate pdf
        for b in zs[:-1]:
            self.survive["pdf"][b] = self.survive["cdf"][b+1] - self.survive["cdf"][b] + 1e-6  # in case of zero
        self.survive["pdf"][zs[-1]] = 1e-6

        return

    def _count(self, data):
        pdf = {}
        cdf = {}

        for d in data:
            price = math.floor(float(d) + epsilon)
            if price in pdf:
                pdf[price] += 1
            else:
                pdf[price] = 1

        for price in range(self.min_price, self.max_price+1):
            if price not in pdf:
                pdf[price] = 0

        for price in pdf:
            pdf[price] = pdf[price]/len(data)

        for price in pdf:
            p = 0
            for j in pdf:
                p += pdf[j] if j <= price else 0
            cdf[price] = p

        return {"pdf": pdf, "cdf": cdf, "data": data}


