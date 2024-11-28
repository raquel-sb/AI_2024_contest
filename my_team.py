# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

##############
# Our Agents #
##############
class OffensiveAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food, taking into account the 3 new features:
    1. Check for the opponent's ghosts -> go to the opposite direction if there is one within 5 squares
    2. After eating 5 food -> go back "home" to obtain points
    3. Try to eat the power capsule
    """
    def __init__(self, index):
        super().__init__(index)
        self.stuck_count = 0
        self.food_carrying = 0
    
    def get_boundary_points(self, game_state):
        """
        Return the x,y-coord of the border
        """
        walls = game_state.get_walls().as_list()
        boundary_x = game_state.data.layout.width // 2 - (2 if self.red else 0)
        return [(boundary_x, y) for x, y in walls if not game_state.has_wall(boundary_x, y)]
    
    def is_in_own_territory(self, game_state, position):
        """
        Verify if the agent is within its own territory
        """
        walls = game_state.get_walls()
        width = walls.width

        mid_x = width // 2 - (2 if self.red else 0) # border x-coord
        if self.red:
            return position[0] <= mid_x  # Red is left side
        else:
            return position[0] > mid_x  # Blue is right side
        
    def get_distance_to_border(self, game_state, position):
        """
        Compute distance to the closest point of its own territory from the current position
        """
        border_positions = self.get_border_positions(game_state)
        distances = [self.get_maze_distance(position, border) for border in border_positions]
        return min(distances) if distances else float('inf')  # Return infinite if there is None

        
    def get_border_positions(self, game_state):
        """
        Find all position on its own territory
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height

        mid_x = width // 2
        border_positions = []
        for y in range(height):
            if self.red:  # If it's red team, search in the left side
                x = mid_x - 2
            else:  # If it's red team, search in the right  side
                x = mid_x

            if not walls[x][y]:  
                border_positions.append((x, y))

        return border_positions
    
    def check_stuck(self, game_state):
        """
        Checks if the agent is stuck in 1 position . Return true if the agent hasn't been moving after 3 states 
        """
        prev_state = self.get_previous_observation()
        if prev_state is None:
            return False
        prev_position = prev_state.get_agent_position(self.index)
        current_position = game_state.get_agent_position(self.index)

        if prev_position == current_position:
            self.stuck_count += 1
        else:
            self.stuck_count = 0  # Reset if the agent moves

        return self.stuck_count >= 3 
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        my_pos = successor.get_agent_state(self.index).get_position()
        my_state = successor.get_agent_state(self.index)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        normal_ghosts = [a for a in ghosts if a.scared_timer == 0] # not scared ghosts
        
        # Reset food carrying count when the agent has returned to its own territory (also when it's killed)
        if self.is_in_own_territory(game_state, my_pos):
            self.food_carrying = 0

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])

            if min_distance == 1:
                self.food_carrying += 1

            features['distance_to_food'] = min_distance
            
        # Return home by going to the nearest territory position 
        boundaries = self.get_boundary_points(game_state)
        min_safe_distance = min([self.get_maze_distance(my_pos, point) for point in boundaries])
        features['distance_to_safe_zone'] = min_safe_distance

        # If ate 3 food, check for path to return home
        if self.food_carrying > 3 * 2:
            distance_to_home = self.get_distance_to_border(game_state, my_pos)
            features['distance_to_home'] = distance_to_home

        # Discourage going near the ghsst if it's within 3 squares
        danger_zone = 3  
        if normal_ghosts:
            ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in normal_ghosts]
            closest_ghost_distance = min(ghost_distances)

            if closest_ghost_distance <= danger_zone:
                features['in_danger'] = 100
                print("Danger!!")
            else:
                features['in_danger'] = 0
        else:
            features['in_danger'] = 0

        # Check for nearby power capsule 
        capsules = self.get_capsules(game_state)
        if len(capsules) > 0:
            min_capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            features['distance_to_food'] = min_capsule_distance * 3

        # Stuck situation
        if self.check_stuck(game_state): 
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance * 10

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -5,
            'distance_to_safe_zone': -0.8, 
            'distance_to_home': -15,  
            'has_food_to_secure': 50,   # Reward for carrying food
            'in_danger': -100,
        }


class DefensiveAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free with 2 more features:
    1. When the oponent's position is None -> go to most recent eaten food position to check for pacman
    2. Avoid the pacman that has eaten our power capsule
    """
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            if my_state.scared_timer == 0:
                features['invader_distance'] = min(dists)
            else:
                features['invader_distance'] = -min(dists) # when the opponet's pacman has eaten the powel capsule, we discourage our agent to get close

        # If the position of the oponent is not found, go for the most recent eaten food
        else:
            # Get food information
            prev_state = self.get_previous_observation()
            food_defending = self.get_food_you_are_defending(game_state).as_list()
            if prev_state:
                prev_food = self.get_food_you_are_defending(prev_state).as_list()
            else:
                prev_food = food_defending

            # Track recently eaten food
            if len(prev_food) > len(food_defending):
                eaten_food = list(set(prev_food) - set(food_defending))
                if len(eaten_food) > 0:
                    eaten_food_pos = eaten_food[0]
                    eaten_food_dist = self.get_maze_distance(my_pos, eaten_food_pos)
                    features['eaten_food_distance'] = eaten_food_dist

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
            
        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 
                'on_defense': 100, 
                'invader_distance': -50, 
                'eaten_food_distance': -80,
                #'food_distance': -5,
                'stop': -100, 
                'reverse': -2
                }
