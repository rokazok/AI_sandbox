import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BanditArm:
    """
    Represents one arm of the multi-armed bandit (e.g., one ad campaign).
    Uses Beta distribution to model the probability of success (click-through).
    
    The Beta distribution is ideal for this as it models probabilities (0-1 range)
    and naturally represents our uncertainty about the true probability.
    """
    name: str
    successes: int = 0  # Number of successful trials (e.g., clicks)
    failures: int = 0   # Number of failed trials (e.g., no clicks)
    
    @property
    def trials(self) -> int:
        """Total number of times this arm has been tried"""
        return self.successes + self.failures
    
    @property
    def mean(self) -> float:
        """
        Current estimate of the arm's success probability (e.g., CTR).
        This is the maximum likelihood estimate: successes / trials.
        """
        if self.trials == 0:
            return 0
        return self.successes / self.trials
    
    def sample_beta(self, size: int = 1) -> np.ndarray:
        """
        Sample from the arm's Beta distribution.
        
        The Beta distribution parameters are:
        - α (alpha) = successes + 1
        - β (beta) = failures + 1
        
        We add 1 to each parameter (Laplace smoothing) to:
        1. Prevent divide-by-zero errors
        2. Start with a uniform prior (Beta(1,1))
        3. Gradually shift from exploration to exploitation as we gather more data
        """
        return np.random.beta(self.successes + 1, self.failures + 1, size=size)

class MultiArmedBandit:
    """
    Implementation of a Multi-Armed Bandit using Thompson Sampling.
    
    Thompson Sampling balances exploration vs exploitation by:
    1. Maintaining a probability distribution for each arm's true success rate
    2. Sampling from these distributions to make probabilistic decisions
    3. Naturally shifting from exploration to exploitation as it gathers more data
    """
    
    def __init__(self, arm_names: List[str]):
        """Initialize a bandit with named arms (e.g., different ad campaigns)"""
        self.arms = {name: BanditArm(name=name) for name in arm_names}
        
    def select_arm(self, n_samples: int = 1000) -> str:
        """
        Use Thompson sampling to select the next arm to try.
        
        Thompson Sampling Process:
        1. For each arm, sample once from its Beta distribution
           - This is equivalent to taking the mean of many samples because
             we're just using it for comparison
           - Sampling just once is much faster and gives similar results
           - The randomness in the single sample still maintains the
             exploration-exploitation balance
        
        This naturally balances exploration vs exploitation because:
        - New/uncertain arms will sometimes sample high values → exploration
        - Well-performing arms will consistently sample high values → exploitation
        
        Note: For visualization purposes, we still use multiple samples in 
        get_probabilities(), but for selection, one sample is sufficient.
        """
        samples = {
            name: arm.sample_beta(size=1)[0]  # Single sample is sufficient
            for name, arm in self.arms.items()
        }
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def update(self, arm_name: str, success: bool) -> None:
        """
        Update the selected arm with the trial result.
        This updates the Beta distribution for the chosen arm.
        """
        arm = self.arms[arm_name]
        if success:
            arm.successes += 1
        else:
            arm.failures += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics for all arms"""
        return {
            name: {
                'trials': arm.trials,
                'successes': arm.successes,
                'mean': arm.mean,
            }
            for name, arm in self.arms.items()
        }
    
    def get_probabilities(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Get current probability distributions for all arms.
        These distributions show our current uncertainty about each arm's true rate.
        - Wider distributions indicate more uncertainty (needs exploration)
        - Narrower distributions indicate more certainty (ready for exploitation)
        """
        return {
            name: arm.sample_beta(size=n_samples)
            for name, arm in self.arms.items()
        }
