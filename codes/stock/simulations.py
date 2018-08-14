# Modules
import numpy as np
import matplotlib.pyplot as plt

# Styles

# Methods

# Classes
## Weighted Majority
class DeterministicWeightedMajority:
    """Deterministic Weighted Majority algorithm implementation.
    """
    # Constructor
    def __init__( self, expert_true, experts_list, epsilon = 0.1, low_threshold = 0.0, high_threshold = 1.0 ):
        # Parameters
        self.predictions = []
        self.epsilon = epsilon
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        ## True
        self.expert_true = expert_true
        self.predictions_true = []
        ## Experts
        self.experts_dict = dict()
        for i, expert in enumerate(experts_list):
            initial_weight = 1 / len( experts_list )
            self.experts_dict[i] = {
                'expert' : experts_list[i],
                'weight' : initial_weight,
                'predictions' : [],
                'weights': [ initial_weight ],
            }
        # Return
        return
    # Renormalize
    def renormalize_weights( self ):
        # Sum of Weights
        weights_sum = 0.0
        for expert_id in self.experts_dict:
            weights_sum += self.experts_dict[expert_id]['weight']
        # Normalize
        for expert_id in self.experts_dict:
            self.experts_dict[expert_id]['weight'] /= weights_sum
        # Return
        return
    # Simulate
    def simulate( self ):
        N_max = len(self.expert_true.series)
        # Run
        for idx in range(1000):
            ## Renormalize
            if( (idx+1) % 1 == 0 ):
                self.renormalize_weights()
            ## Collect Votes
            votes_info = dict()
            for expert_id in self.experts_dict:
                ### Expert Info
                expert_dict = self.experts_dict[expert_id]
                expert_advice = expert_dict['expert'].get( idx )
                expert_weight = expert_dict['weight']
                expert_dict['predictions'].append( expert_advice )
                ### Collect Vote
                if( expert_advice in votes_info ):
                    votes_info[expert_advice] += expert_weight
                else:
                    votes_info[expert_advice] = expert_weight
            ## Predict
            assert len(votes_info) > 0, "No votes submitted by experts!?"
            selected_vote, selected_weight = None, -1
            for vote in votes_info:
                if( votes_info[vote] > selected_weight ):
                    selected_vote = vote
                    selected_weight = votes_info[vote]
            ### Store Prediction
            self.predictions.append( selected_vote )
            ## Suffer
            ### True Prediction
            prediction_true = self.expert_true.get( idx )
            self.predictions_true.append( prediction_true )
            ### Loss Function
            for expert_id in self.experts_dict:
                ### Expert Info
                expert_dict = self.experts_dict[expert_id]
                expert_advice = expert_dict['predictions'][-1]
                expert_weight = expert_dict['weight']
                ### Update Weight
                expert_weight *= (1 - self.epsilon)**abs(prediction_true - expert_advice)
                expert_weight = min( self.high_threshold, max( self.low_threshold, expert_weight ) ) # Apply Thresholds
                expert_dict['weight'] = expert_weight
                expert_dict['weights'].append( expert_weight )
        # Return
        return
    # Visualize
    def visualize( self ):
        # Create new Figure
        plt.figure( figsize=(16,10) )
        # Thresholds
        ## High
        plt.axhline( self.high_threshold , color='black' )
        ## Dominance
        plt.axhline( 0.5 , color='red' )
        ## Low
        plt.axhline( self.low_threshold , color='black' )
        # Create Data
        N_experts = len(self.experts_dict)
        N_weights = len(self.experts_dict[0]['weights'])
        weight_ratios = np.zeros( (N_experts, N_weights) )
        ## Get Weights
        for idx in range(N_experts):
            weight_ratios[idx] = self.experts_dict[idx]['weights']
        ## Normalize Columns
        weight_ratios /= np.sum( weight_ratios, axis = 0 )
        # Add Charts
        expert_names = [ self.experts_dict[idx]['expert'].name for idx in range(N_experts) ]
        ## Plot
        for idx in range(N_experts):
            plt.plot( weight_ratios[idx], label = expert_names[idx] )
        # Show
        plt.title( 'Ratio of Weights for Different Experts' )
        plt.legend()
        plt.show()
        # Return
        return





























