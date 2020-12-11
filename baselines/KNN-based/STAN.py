import math
from _operator import itemgetter


class STAN:
    def __init__(self, session_id, session, session_timestamp, sample_size=0, k=500, factor1=True, l1=3.54,
                 factor2=True, l2=20 * 24 * 3600, factor3=True, l3=3.54):
        self.k = k
        self.sample_size = sample_size
        self.session_all = session
        self.session_id_all = session_id
        self.session_timestamp_all = session_timestamp
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        # cache
        self.session_timestamp_cache = {}  # session_id: timestamp
        self.item_session_cache = {}  # item_id: set(session_id)
        self.session_item_cache = {}  # session_id: set(item_id)
        self.session_item_list_cache = {}  # session_id: [item_id]
        for i, session in enumerate(self.session_all):
            sid = self.session_id_all[i]  # current session id
            self.session_timestamp_cache.update({sid: self.session_timestamp_all[i]})
            for item in session:
                session_map = self.item_session_cache.get(item)
                if session_map is None:
                    session_map = set()
                    self.item_session_cache.update({item: session_map})
                session_map.add(sid)

                item_map = self.session_item_cache.get(sid)
                if item_map is None:
                    item_map = set()
                    self.session_item_cache.update({sid: item_map})
                item_map.add(item)

                item_list = self.session_item_list_cache.get(sid)
                if item_list is None:
                    item_list = []
                    self.session_item_list_cache.update({sid: item_list})
                item_list += [item]

        self.current_session_weight_cache = {}
        self.current_timestamp = 0

    def find_neighbours(self, session_items, input_item):
        # neighbour candidate
        possible_neighbours = self.possible_neighbour_sessions(session_items, input_item)
        # get top k according to similarity
        possible_neighbours = self.cal_similarity(session_items, possible_neighbours)
        possible_neighbours = sorted(possible_neighbours, reverse=True, key=lambda x: x[1])
        possible_neighbours = possible_neighbours[:self.k]

        return possible_neighbours

    def possible_neighbour_sessions(self, session_items, input_item):
        neighbours = set()
        for item in session_items:
            if item in self.item_session_cache:
                neighbours = neighbours | self.item_session_cache.get(item)
        return neighbours

    # find recent session (not use)
    def most_recent_sessions(self, sessions):
        recent_session = set()
        tuples = []
        for session in sessions:
            time = self.session_timestamp_cache.get(session)
            tuples += [(session, time)]

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        cnt = 0
        for i in tuples:
            cnt += 1
            if cnt > self.sample_size:
                break
            recent_session.add(i[0])
        return recent_session

    # calculate similarity
    def cal_similarity(self, session_items, sessions):
        neighbours = []
        for session in sessions:
            neighbour_session_items = self.session_item_cache.get(session)
            similarity = self.cosine_similarity(neighbour_session_items, session_items)
            if self.factor2 is True:
                similarity = similarity * math.exp(
                    -(self.current_timestamp - self.session_timestamp_cache[session]) / self.l2)
            if similarity > 0:
                neighbours += [(session, similarity)]
        return neighbours

    # cosine similarity
    def cosine_similarity(self, s1, s2):
        common_item = s1 & s2
        similarity = 0
        for item in common_item:
            similarity += self.current_session_weight_cache[item]
        l1 = len(s1)
        l2 = len(s2)
        return similarity / math.sqrt(l1 * l2)

    def score_items(self, neighbours, session_items):
        scores = {}
        for session in neighbours:
            items = self.session_item_cache.get(session[0])

            common_items_idx = -1
            for i, item in enumerate(items):
                if item in session_items:
                    common_items_idx = i

            for idx, item in enumerate(items):
                old_score = scores.get(item)
                if self.factor3 is True:
                    new_score = math.exp(-math.fabs(idx - common_items_idx) / self.l3) * session[1]
                else:
                    new_score = session[1]

                if old_score is None:
                    scores.update({item: new_score})
                else:
                    new_score += old_score
                    scores.update({item: new_score})
        return scores

    def predict(self, session_id, session_items, session_timestamp, k=20):
        # initialize
        length = len(session_items)
        for idx, item in enumerate(session_items):
            if self.factor1 is True:
                weight = math.exp((idx + 1 - length) / self.l1)
            else:
                weight = 1
            if item in self.current_session_weight_cache:
                self.current_session_weight_cache.update({item: max(weight, self.current_session_weight_cache[item])})
            else:
                self.current_session_weight_cache.update({item: weight})
        self.current_timestamp = session_timestamp

        last_item_id = session_items[-1]
        neighbours = self.find_neighbours(set(session_items), last_item_id)
        scores = self.score_items(neighbours, session_items)
        scores_sorted_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return scores_sorted_list
