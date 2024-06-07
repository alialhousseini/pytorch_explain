import abc
import torch
from torch.nn import functional as F


class Logic:
    @abc.abstractmethod
    def update(self):
        """Abstract method to update the internal state, if necessary."""
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, a, dim=1):
        """Abstract method to compute the conjunction (AND) over a tensor along a specified dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a, dim=1):
        """Abstract method to compute the disjunction (OR) over a tensor along a specified dimension."""
        raise NotImplementedError

    def conj_pair(self, a, b):
        """Abstract method to compute the conjunction (AND) between two tensors."""
        raise NotImplementedError

    def disj_pair(self, a, b):
        """Abstract method to compute the disjunction (OR) between two tensors."""
        raise NotImplementedError

    def iff_pair(self, a, b):
        """Abstract method to compute the biconditional (IFF) between two tensors."""
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        """Abstract method to compute the negation (NOT) of a tensor."""
        raise NotImplementedError


class ProductTNorm(Logic):
    """
    Implements boolean algebra transformations using Product T-Norm logic.

    This class uses product-based operations to represent logical conjunction, disjunction, and negation:
    - Conjunction (AND) is represented by the product of values.
    - Disjunction (OR) is represented by the formula: 1 - product(1 - values).
    - Negation (NOT) is represented by 1 - value.
    """

    def __init__(self):
        super(ProductTNorm, self).__init__()
        self.current_truth = torch.tensor(1)
        self.current_false = torch.tensor(0)

    def update(self):
        pass

    def conj(self, a, dim=1):
        """Compute conjunction (AND) using product along a specified dimension."""
        return torch.prod(a, dim=dim, keepdim=True)

    def conj_pair(self, a, b):
        """Compute conjunction (AND) between two tensors using product."""
        return a * b

    def disj(self, a, dim=1):
        """Compute disjunction (OR) using the formula: 1 - product(1 - values) along a specified dimension."""
        return 1 - torch.prod(1 - a, dim=dim, keepdim=True)

    def disj_pair(self, a, b):
        """Compute disjunction (OR) between two tensors using the formula: a + b - a * b."""
        return a + b - a * b

    def iff_pair(self, a, b):
        """Compute biconditional (IFF) between two tensors using conjunction of disjunctions."""
        return self.conj_pair(self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        """Compute negation (NOT) as 1 - value."""
        return 1 - a

    def predict_proba(self, a):
        """Predict probabilities by squeezing the last dimension."""
        return a.squeeze(-1)


class GodelTNorm(Logic):
    """
    Implements boolean algebra transformations using Godel T-Norm logic.

    This class uses min/max operations to represent logical conjunction, disjunction, and negation:
    - Conjunction (AND) is represented by the minimum value.
    - Disjunction (OR) is represented by the maximum value.
    - Negation (NOT) is represented by 1 - value.
    """

    def __init__(self):
        super(GodelTNorm, self).__init__()
        self.current_truth = 1
        self.current_false = 0

    def update(self):
        pass

    def conj(self, a, dim=1):
        """Compute conjunction (AND) using the minimum value along a specified dimension."""
        return torch.min(a, dim=dim, keepdim=True)[0]

    def disj(self, a, dim=1):
        """Compute disjunction (OR) using the maximum value along a specified dimension."""
        return torch.max(a, dim=dim, keepdim=True)[0]

    def conj_pair(self, a, b):
        """Compute conjunction (AND) between two tensors using the minimum value."""
        return torch.minimum(a, b)

    def disj_pair(self, a, b):
        """Compute disjunction (OR) between two tensors using the maximum value."""
        return torch.maximum(a, b)

    def iff_pair(self, a, b):
        """Compute biconditional (IFF) between two tensors using conjunction of disjunctions."""
        return self.conj_pair(self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b)))

    def neg(self, a):
        """Compute negation (NOT) as 1 - value."""
        return 1 - a

    def predict_proba(self, a):
        """Predict probabilities by squeezing the last dimension."""
        return a.squeeze(-1)


if __name__ == "__main__":
    # Create example tensors for testing
    a = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    b = torch.tensor([[0.3, 0.7], [0.8, 0.2]])

    # Initialize logic objects
    product_tnorm = ProductTNorm()
    godel_tnorm = GodelTNorm()

    # Test Product T-Norm logic operations
    print("Product T-Norm Logic:")
    print("Conjunction (AND):", product_tnorm.conj(a))
    print("Disjunction (OR):", product_tnorm.disj(a))
    print("Negation (NOT):", product_tnorm.neg(a))
    print("IFF:", product_tnorm.iff_pair(a, b))

    # Test Godel T-Norm logic operations
    print("\nGodel T-Norm Logic:")
    print("Conjunction (AND):", godel_tnorm.conj(a))
    print("Disjunction (OR):", godel_tnorm.disj(a))
    print("Negation (NOT):", godel_tnorm.neg(a))
    print("IFF:", godel_tnorm.iff_pair(a, b))
