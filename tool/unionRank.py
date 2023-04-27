import numpy as np
from collections import defaultdict


class RankAggregator(object):
    """
    Generate rank aggregator for partial rank method.
    """
    def __init__(self):
        pass


    def item_universe(self,rank_list):
        """
        Determines the universe of ranked items (union of all the items ranked by all
        experts
        """
        return list(frozenset().union(*[list(x.keys()) for x in rank_list]))


    def first_order_marginals(self,rank_list):
        """
        Computes m_ik, the fraction of rankers that ranks item i as their kth choice
        (see Ammar and Shah, "Efficient Rank Aggregation Using Partial Data").  Works
        with either full or partial lists.
        """
        # get list of all the items
        all_items = self.item_universe(rank_list)
        # dictionaries for creating the matrix
        self.item_mapping(all_items)
        # create the m_ik matrix and fill it in
        m_ik = np.zeros((len(all_items),len(all_items)))
        n_r = len(rank_list)
        for r in rank_list:
            for item in r:
                m_ik[self.itemToIndex[item],r[item]-1] += 1
        return m_ik/n_r


    def convert_to_ranks(self,scoreDict):
        """
        Accepts an input dictionary in which they keys are items to be ranked (numerical/string/etc.)
        and the values are scores, in which a higher score is better.  Returns a dictionary of
        items and ranks, ranks in the range 1,...,n.
        """
        # default sort direction is ascending, so reverse (see sort_by_value docs)
        x = np.sort_by_value(scoreDict,True)
        y = list(zip(list(zip(*x))[0],range(1,len(x)+1)))
        ranks = {}
        for t in y:
            ranks[t[0]] = t[1]
        return ranks


    def item_ranks(self,rank_list):
        """
        Accepts an input list of ranks (each item in the list is a dictionary of item:rank pairs)
        and returns a dictionary keyed on item, with value the list of ranks the item obtained
        across all entire list of ranks.
        """
        item_ranks = {}.fromkeys(rank_list[0])
        for k in item_ranks:
            item_ranks[k] = [x[k] for x in rank_list]
        return item_ranks


    def item_mapping(self,items):
        """
        Some methods need to do numerical work on arrays rather than directly using dictionaries.
        This function maps a list of items (they can be strings, ints, whatever) into 0,...,len(items).
        Both forward and reverse dictionaries are created and stored.
        """
        self.itemToIndex = {}
        self.indexToItem = {}
        indexToItem = {}
        next = 0
        for i in items:
            self.itemToIndex[i] = next
            self.indexToItem[next] = i
            next += 1
        return


class PartialRank(RankAggregator):
    """
    Run partial rank aggregation. 
    """

    def __init__(self):
        super(RankAggregator, self).__init__()
        self.mDispatch = {'borda': self.borda_aggregation,
                          'modborda': self.modified_borda_aggregation,
                          'lone': self.lone_aggregation}


    def aggregate_ranks(self, experts, method='borda', stat='mean'):
        """
        Combines the ranks in the list experts to obtain a single set of aggregate ranks.
        Currently operates only on ranks, not scores.

        INPUT:
        ------
            experts: list of dictionaries, required
                each element of experts should be a dictionary of item:rank
                pairs

            method: string, optional
                which method to use to perform the rank aggregation

            stat: string, optional
                statistic used to combine Borda scores; only relevant for Borda
                aggregation
        """
        aggregated_ranks = {}
        scores = {}
        if method in self.mDispatch:
            if ['borda'].count(method) > 0:
                # convert to truncated Borda lists
                supp_experts = self.supplement_experts(experts)
                scores, aggranks = self.mDispatch['borda'](supp_experts,stat)
            else:
                # methods not supplement expert lists
                scores, aggranks = self.mDispatch[method](experts)
        else:
            print('ERROR: method', method, 'invalid.')
        return scores, aggregated_ranks


    def supplement_experts(self, experts):
        """
        Converts partial lists to full lists by supplementing each expert's ranklist with all
        unranked items, each item having rank max(rank) + 1 (different for different experts).
        (This has the effect of converting partial Borda lists to full lists via truncated
        count).
        """
        supp_experts = []
        # get the list of all the items
        all_items = self.item_universe(experts)
        for rank_dict in experts:
            new_ranks = {}
            max_rank = max(rank_dict.values())
            for item in all_items:
                if item in rank_dict:
                    new_ranks[item] = rank_dict[item]
                else:
                    new_ranks[item] = max_rank + 1
            supp_experts.append(new_ranks)
        return supp_experts


    def borda_aggregation(self,supp_experts,stat):
        """
        Rank is equal to mean rank on the truncated lists; the statistic is also returned,
        since in a situation where rankers only rank a few of the total list of items, ties are
        quite likely and some ranks will be arbitrary.

        Choices for the statistic are mean, median, and geo (for geometric mean).
        """
        scores = {}
        stat_dispatch = {'mean':np.mean,'median':np.median,'geo':np.gmean}
        # all lists are full, so any set of dict keys will do
        for item in supp_experts[0].keys():
            vals_list = [x[item] for x in supp_experts]
            scores[item] = stat_dispatch[stat](vals_list)
        # in order to use convert_to_ranks, we need to manipulate the scores
        #   so that higher values = better; right now, lower is better.  So
        #   just change the sign of the score
        flip_scores = np.copy.copy(scores)
        for k in flip_scores:
            flip_scores[k] = -1.0*flip_scores[k]
        agg_ranks = self.convert_to_ranks(flip_scores)
        return scores,agg_ranks


    def modified_borda_aggregation(self,experts):
        """
        Uses modified Borda counts to deal with partial lists.  For a ranker who only
        ranks m < n options, each item recieves a score of max(m + 1 - r,0), where r is the
        rank of the item.  This has the effect of giving the last item ranked m points
        and any unranked items zero points.  Ranks are then set using the modified
        scores.
        """
        scores = defaultdict(int)
        # lists are not full, so we need the universe of ranked items
        all_items = self.item_universe(experts)
        for ranker in experts:
            m = len(ranker)
            for item in ranker:
                scores[item] += m + 1 - ranker[item]
        # now convert scores to ranks
        agg_ranks = self.convert_to_ranks(scores)
        return scores,agg_ranks


    def lone_aggregation(self,experts):
        """
        Implements the l1 ranking scheme of Ammar and Shah, "Efficient Rank
        Aggregation Using Partial Data"
        """
        scores = {}
        # construct the ranker-item matrix m_ik
        m_ik = self.first_order_marginals(experts)
        n = len(self.itemToIndex)
        # now do the multiplication
        s_vec = n - np.array(list(range(1,n+1)))
        s_vec = np.dot(m_ik,s_vec)
        # array of scores
        for i in range(len(s_vec)):
            scores[self.indexToItem[i]] = s_vec[i]
        # convert to ranks
        agg_ranks = self.convert_to_ranks(scores)
        return scores,agg_ranks