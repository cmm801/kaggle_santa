import pandas as pd
import numpy as np
import datetime
from collections import deque
import matplotlib.pyplot as plt

N_DAYS = 100
N_CHOICES = 10
N_FAMILIES = 5000
MAX_FAMILY_SIZE = 8
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
MAX_PREFERENCE_COST = 3972

class AssignmentHelper():

    def __init__(self):
        input_data = pd.read_csv( '../input/family_data.csv', index_col='family_id')

        # Get preferences and family sizes from input information
        self.preferences = input_data.drop(['n_people'], axis=1).to_numpy() - 1        
        self.family_sizes = input_data['n_people'].to_numpy()
 
        # Initialize a matrix for preference costs (# Families x # Days)
        # Only will be populated if and when get_preference_cost_matrix is called
        self.pref_cost_matrix = None
    
        # Calculate preference costs according to the family size
        # Each row index corresponds to the size of the family (i.e., row index 4 is a 4-person family)        
        self.pref_cost_lookup = np.ones( (MAX_FAMILY_SIZE + 1, N_CHOICES + 1), dtype=np.int32 )
        for fsize in range(MAX_FAMILY_SIZE + 1):
            self.pref_cost_lookup[fsize,:] = self.calc_preference_cost_single(fsize)                    

    def get_assignment_rank( self, assignment, prefs=None):
        """Get the rank (in preferences) of each assignment
           If the assignment is not in the family's preferences, then the rank is N_CHOICES.
           By default, the input is the full array of preferences.
           Input can also be a sub-assignment and the corresponding sub-preferences for some family(ies)."""
            
        if prefs is None:
            prefs = self.preferences
            
        assignment_rank = N_CHOICES * np.ones( (prefs.shape[0],), dtype=np.int32 )
        assert prefs.shape[1] == N_CHOICES, 'Incorrect dimenion for input preferences.'

        if isinstance(assignment, list):
            assignment = np.array(assignment)    
            
        idx_T = np.where( assignment == prefs )
        idx = np.vstack(idx_T).T
        if idx.size:
            assignment_rank[idx[:,0]] = idx[:,1]
        return assignment_rank    

    def get_total_cost(self, assignment):
        """Get the total cost (preference cost + accounting cost) for a given assignment.
        """
        occupancy = self.calc_occupancy(assignment)
        preference_cost = self.calc_preference_cost(assignment)
        accounting_cost = self.calc_accounting_cost(occupancy)
        return preference_cost.sum() + accounting_cost.sum()
            
    def calc_occupancy(self, assignment):
        """Calculate the occupancy from an assignment and the family sizes.
        """
        occupancy = np.zeros( (N_DAYS,), dtype=np.int32)
        for family_id, assigned_day in enumerate(assignment):
            occupancy[assigned_day] += self.family_sizes[family_id]
        return occupancy    
                
    def calc_occupancy_prob(self, assignment_prob):
        """Calculate the occupancy from a probabilistic assignment and the family sizes.
        """
        occupancy = ( assignment_prob * helper.family_sizes[:,np.newaxis] ).sum(axis=0)        
        return occupancy        
    
    def calc_preference_cost(self, assignment):
        """Calculates the preference cost of a complete assignment.
        Returns an array of length N_FAMILIES, attributing the costs to individual assignments.
        """    
        # Get the rank (in preferences) of each assignment
        # If the assignment is not in the family's preferences, then the rank is N_CHOICES
        assignment_rank = self.get_assignment_rank( assignment)

        # Get the vector of costs according to the assignment rank
        return self.pref_cost_lookup[ self.family_sizes, assignment_rank ]
    
    def get_preference_penalty(self, assignment_prob, idx=None):
        """Get the penalty size to be assessed on assignments when they violate constraints.
        """
        if idx is None:
            return np.max( self.pref_cost_matrix, axis=1, keepdims=True)
        else:
            return self.pref_cost_matrix[idx,:].max()
        
    def calc_preference_cost_prob(self, assignment_prob, idx=None):
        """Calculates the preference cost based on probabilities of assignment. 
        Adds a penalty if the individual preferences do not sum to one, or if they are outside of [0,1].
        """
        A = assignment_prob        
        if idx is None:
            assert A.shape[0] == N_FAMILIES, 'Dimensional mismatch.'
            PC = A * self.get_preference_cost_matrix() 
        else:
            assert A.shape[0] == 1, 'Dimensional mismatch.'            
            PC = A * self.get_preference_cost_matrix()[idx,:]
            
        pref_penalty = self.get_preference_penalty(A, idx=idx)
    
        # Add penalty for assignment being outside of the interval [0,1]
        PC += np.maximum(0, -A) * pref_penalty
        PC += np.maximum(0, A - 1) * pref_penalty
        
        # Add penalty for assignment rows not summing to 1
        PC += np.abs(1 - A.sum(axis=1,keepdims=True) ) * pref_penalty / N_DAYS
        return PC    
    
    def calc_accounting_cost(self, occupancy):
        """Calculates the accounting cost based on the occupancy on each day.
        Returns an array of length N_DAYS, attributing costs to individual daily occupancies."""

        # Get the difference between occupancy on consecutive days
        abs_diff = np.abs( occupancy[:-1] - occupancy[1:] )

        # INITIAL CONDITION - specifies the difference on the start day (day 100) is defined to be 0
        abs_diff = np.hstack( [ abs_diff, [0] ] )
        
        # Use the formula for the accounting cost, and floor at 0
        acc_cost = (occupancy - MIN_OCCUPANCY) / 400.0 * occupancy ** (0.5 + abs_diff / 50.0)
        acc_cost = np.maximum( 0, acc_cost )
            
        # Add penalty for occupancy being lower than 125 or higher than 300
        occupancy_penalty = self.get_occupancy_penalty_per_person()        
        acc_cost += np.maximum(0, MIN_OCCUPANCY - occupancy) * occupancy_penalty
        acc_cost += np.maximum(0, occupancy - MAX_OCCUPANCY) * occupancy_penalty
        
        return acc_cost

    def get_occupancy_penalty_per_person(self):
        return np.mean( self.pref_cost_lookup[1:,-1] / np.arange(1, MAX_FAMILY_SIZE + 1) )                
    
    def calc_total_cost(self, assignment, occupancy=None ):
        """Calculate the total cost of an assignment.
        """
        if occupancy is None:
            occupancy = np.zeros( (N_FAMILIES,), dtype=np.int32)
            for family_id, assigned_day in enumerate(assignment):
                occupancy[assigned_day] += self.family_sizes[family_id]
                
        preference_cost = self.calc_preference_cost(assignment)
        accounting_cost = self.calc_accounting_cost(occupancy)
        return np.sum(preference_cost) + np.sum(accounting_cost)    

    def calc_preference_cost_single(self, n):
        """Calculate the preference cost for a single family of size n.
        Returns an array of size 11. 
        The first 10 entries represent the costs of choices 1-10.
        The last entry represents the cost of a non-assignment.
        """
        # Cost for being assigned to choices 1-10
        pref_cost = np.ones( (11,), dtype=np.int32 )
        pref_cost[0] = 0
        pref_cost[1] = 50
        pref_cost[2] = 50 + 9 * n
        pref_cost[3] = 100 + 9 * n
        pref_cost[4] = 200 + 9 * n
        pref_cost[5] = 200 + 18 * n
        pref_cost[6] = 300 + 18 * n
        pref_cost[7] = 300 + 36 * n
        pref_cost[8] = 400 + 36 * n
        pref_cost[9] = 500 + 36 * n + 199 * n

        # Penalty for being unassigned to a day
        pref_cost[10] = 500 + 36 * n + 398 * n    
        return pref_cost
    
    def calc_accounting_cost_for_day_change(self, occupancy, family_id, curr_day, new_day ):
        """ Calculate the accounting cost of moving a family from their current assignment to a new assignment.
        """
        fam_size = self.family_sizes[family_id]   
        cost_of_rem = self.calc_accounting_cost_for_occ_chg( occupancy, curr_day, -fam_size )
        cost_of_add = self.calc_accounting_cost_for_occ_chg( occupancy, new_day, +fam_size )
        return cost_of_add + cost_of_rem    

    def calc_preference_cost_for_rank_change( self, family_id, curr_rank, new_rank ):
        """Calculate the preference cost of moving a family from their M-th choice to their N-th choice.
        """
        family_size = self.family_sizes[family_id]
        pref_costs = self.pref_cost_lookup[family_size,:]
        preference_cost_change = pref_costs[new_rank] - pref_costs[curr_rank]
        return preference_cost_change    

    def calc_total_cost_change_by_day( self, occupancy, family_id, curr_day, new_day ):
        """Calculate the total cost of moving a family from their current day to another.
        """    
        curr_rank = self.get_assignment_rank( [curr_day], self.preferences[[family_id],:])
        new_rank = self.get_assignment_rank( [new_day], self.preferences[[family_id],:])

        preference_cost = self.calc_preference_cost_for_rank_change( family_id, curr_rank, new_rank )
        accounting_cost = self.calc_accounting_cost_for_day_change( occupancy, family_id, curr_day, new_day )
        return preference_cost + accounting_cost

    def calc_accounting_cost_for_occ_chg( self, occupancy, day, chg_amt ):
        """Calculate the change in the accounting cost if the occupancy of a given day changes.
        The 'day' argument specifies [0,99] for the # of days before Christmas.
        The 'chg_amt' argument specifes the increase in occupancy for positive values, 
           or the decrease in occupancy for negative values on the given day."""

        if 0 < day:
            sub_occ = occupancy[day-1:day+2].copy()
            idx_day = 1
        else:
            sub_occ = occupancy[day:day+2].copy()
            idx_day = 0

        # Calculate the original accounting cost
        orig_cost = np.sum( self.calc_accounting_cost(sub_occ) )

        # Calculate the new cost with the new occupancy on 'day'
        sub_occ[idx_day] += chg_amt
        new_cost = np.sum( self.calc_accounting_cost(sub_occ) )

        return new_cost - orig_cost            

    def get_preference_rank_matrix(self):
        """Return a matrix of size (# Families) x (# Days) containing the family's ranked preference for that day.
        """
        pref_rank = N_CHOICES * np.ones( (N_FAMILIES, N_DAYS), dtype=np.int32 )
        for j in range(N_CHOICES):
            pref_rank[( np.arange(N_FAMILIES), self.preferences[:,j] ) ] = j
        return pref_rank
    
    def get_preference_cost_matrix(self):
        """Return a matrix of size (# Families) x (# Days) containing the cost of each family being assigned any day.
        """
        if self.pref_cost_matrix is None:
            pref_rank = self.get_preference_rank_matrix()
            pref_costs_lookup_expanded = self.pref_cost_lookup[ self.family_sizes ]
            row_idx = np.tile( np.arange(N_FAMILIES).reshape(N_FAMILIES,1), N_DAYS )
            self.pref_cost_matrix = pref_costs_lookup_expanded[row_idx,pref_rank]        
        return self.pref_cost_matrix

    
class AssignmentManager():
    
    def __init__(self):

        self.helper = AssignmentHelper()
        self.preferences = self.helper.preferences
        self.family_sizes = self.helper.family_sizes
        self.pref_cost_lookup = self.helper.pref_cost_lookup
    
    def set_assignment(self, assignment):
        self.assignment = assignment.copy()
        self.occupancy = self.helper.calc_occupancy(assignment)
    
    def get_accounting_cost(self):
        return self.helper.calc_accounting_cost(self.occupancy )

    def get_preference_cost(self):
        return self.helper.calc_preference_cost(self.assignment )
        
    def get_assignment_rank(self):
        return self.helper.get_assignment_rank( self.assignment )
            
    def get_total_cost(self):
        return np.sum( self.get_preference_cost() ) + np.sum( self.get_accounting_cost() )
    
    def calc_total_cost_change_by_day( self, family_id, new_day):
        curr_day = int(self.assignment[family_id])
        return self.helper.calc_total_cost_change_by_day( self.occupancy, family_id, curr_day, new_day)
    
    def move_family(self, family_id, new_day ):
        curr_day = int(self.assignment[family_id])
        family_size = self.family_sizes[family_id]

        self.occupancy[curr_day] -= family_size
        self.occupancy[new_day] += family_size
        self.assignment[family_id] = new_day
    
    def anneal(self, ran_gen, T_init=100, cooling=0.95 ):
        """Run simulated annealing"""

        T = T_init
        changes = deque(maxlen=10)
        last_cost = self.get_total_cost() - 1000
        ctr = 0
        while True:
            if ctr % 1000 == 0:
                curr_cost = self.get_total_cost()                
                if ctr > 0:
                    print( [ ctr, T, curr_cost ] )
                    T *= cooling
                    changes.appendleft( curr_cost - last_cost )
                    if T < 5 and len(changes) >= 10 and np.mean(changes) > -100:
                        return
                last_cost = self.get_total_cost()

            idx_family = ran_gen.choice(N_FAMILIES)
            curr_day = self.assignment[idx_family]

            # loop over each family choice
            for idx_rank in range(N_CHOICES):
                new_day = ran_gen.choice(N_DAYS)
                chg_cost = self.calc_total_cost_change_by_day(idx_family, new_day)
                if chg_cost < 0 or np.exp( -chg_cost/T ) > ran_gen.rand():
                    self.move_family(idx_family, new_day )                
            ctr += 1                       

    def load_from_csv(self, file_name):
        """Load a saved assignment from csv.
        """
        print(file_name)
        assignment_pd = pd.read_csv( '../assignments/' + file_name, index_col='family_id')        
        
        # Subtract one to convert our day count into the Kaggle convention        
        assignment = assignment_pd.to_numpy() - 1        
        self.set_assignment(assignment)
            
    def save_to_csv(self):
        """Save a submission to a csv file. Use the old submission file as a template."""
        
        new_index = pd.Index( np.arange(N_FAMILIES), name='family_id')
        new_sub = pd.DataFrame( 1 + self.assignment, columns=['assigned_day'], index=new_index )
        
        score = int(self.get_total_cost())
        curr_date = datetime.datetime.date(datetime.datetime.now())
        new_sub.to_csv( '../assignments/' + f'submission_{score}_{curr_date}.csv')
        
        

class ProbAM():
    """Probabilistic assignment manager. Each assignment is a probability distribution over all days."""
    
    def __init__(self):

        self.helper = AssignmentHelper()
        self.pref_rank_mtx = helper.get_preference_rank_matrix()
        
        # Initialize properties that will be set when the initial assignment is provided
        self.assignment_prob = None
        self.occupancy_prob = None
        self.accounting_cost = None
    
    def set_assignment(self, assignment_prob):
        self.assignment_prob = assignment_prob.copy()
        self.occupancy_prob = ( assignment_prob * helper.family_sizes[:,np.newaxis] ).sum(axis=0)        

        self._accounting_cost = self.get_accounting_cost()
        self._preference_cost = self.get_preference_cost().sum(axis=1)
    
    def get_accounting_cost(self):
        return self.helper.calc_accounting_cost(self.occupancy_prob )
                    
    def get_preference_cost(self, idx=None):
        return self.helper.calc_preference_cost_prob( self.assignment_prob, idx=idx)
            
    def get_total_cost(self):
        return np.sum( self.get_preference_cost() ) + np.sum( self.get_accounting_cost() )
    
    def get_pref_cost_grad(self, family_id=None):
        if family_id is not None:
            grads = self.pref_cost_matrix[[family_id],:]
        else:
            grads = self.pref_cost_matrix
            
        # Adjust the gradients so that each row of the assignment probability sums to 1
        grads[:-1] = np.hstack( [ grads[:,:-1] - grads[:,-1], \
                          np.sum( grads[:,-1] - grads[:,:-1], axis=1) ] )
        
    def get_acct_cost_grad(self, family_id=None):
        pass

    def calc_acct_cost_change(self, idx_family, chg_occ ):        
        curr_acct_cost = self._accounting_cost
        new_acct_cost = self.helper.calc_accounting_cost( self.occupancy_prob + chg_occ )
        return new_acct_cost - curr_acct_cost
        
    def calc_pref_cost_change(self, idx_family, chg_vec):        
        A = self.assignment_prob[ [idx_family],:]
        curr_pref_cost = self.helper.calc_preference_cost_prob( A, idx=idx_family)
        new_pref_cost = self.helper.calc_preference_cost_prob( A + chg_vec, idx=idx_family)
        return new_pref_cost.ravel() - curr_pref_cost.ravel()
        
    def calc_total_cost_change(self, idx_family, chg_vec):
        pref_cost_chg = self.calc_pref_cost_change(idx_family, chg_vec)
        acct_cost_chg = self.calc_acct_cost_change(idx_family, chg_vec)        
        return pref_cost_chg + acct_cost_chg
        
    def _get_change_vector(self, idx_family, idx_day, chg_scale=0.05):
        chg_vec = np.zeros( (N_DAYS,))
        chg_vec[idx_day] = chg_scale
        chg_vec -= chg_vec.mean()
        
        # Make sure the resulting assignments will be in [0,1]
        tot = self.assignment_prob[idx_family,:] + chg_vec
        tot = np.clip( tot, 1e-10, 1 - 1e-10 )
        return tot - self.assignment_prob[idx_family,:]
        
        return chg_vec
    
    def anneal(self, ran_gen, T_init=10, cooling=0.95 ):
        """Run simulated annealing"""

        T = T_init
        changes = deque(maxlen=10)
        last_cost = self.get_total_cost() - 1000
        ctr = 0
        while True:
            if ctr % 1000 == 0:
                curr_cost = self.get_total_cost()                
                if ctr > 0:
                    print( [ ctr, T, curr_cost ] )
                    T *= cooling
                    changes.appendleft( curr_cost - last_cost )
                    if T < T_init/100 and len(changes) >= 10 and np.mean(changes) > -1:
                        return
                last_cost = self.get_total_cost()

            # Choose a random family, weighted by how high their preference cost is
            fam_prob = self._preference_cost / self._preference_cost.sum()
            idx_family = ran_gen.choice(N_FAMILIES, p=fam_prob)

            # Move some probability mass to a single day. Weight the change by the inverse pref cost matrix
            adj_pref_costs = pam.helper.get_preference_cost_matrix()[idx_family,:]
            inv_pref_costs = ( 20 + MAX_PREFERENCE_COST - adj_pref_costs ) ** 2
            new_day = ran_gen.choice( N_DAYS, p=inv_pref_costs/inv_pref_costs.sum())

            # Get the change vector
            chg_scale = ran_gen.beta(10,40)
            chg_prob = self._get_change_vector(idx_family, new_day, chg_scale)
            chg_occ = chg_prob * self.helper.family_sizes[idx_family]

            # Choose whether or not to accept the change
            chg_acct_cost = self.calc_acct_cost_change(idx_family, chg_occ )
            chg_pref_cost = self.calc_pref_cost_change(idx_family, chg_prob )            
            chg_tot_cost = chg_acct_cost.sum() + chg_pref_cost.sum() 
            if chg_tot_cost < 0 or np.exp( -chg_tot_cost/T ) > ran_gen.rand():
                self.assignment_prob[idx_family,:] += chg_prob
                self.occupancy_prob += chg_occ
                self._accounting_cost += chg_acct_cost
                self._preference_cost[idx_family] += chg_pref_cost.sum()       
            ctr += 1                       
        
