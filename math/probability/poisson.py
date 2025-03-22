#!/usr/bin/env python3

class Poisson:
    """
    Classe représentant une distribution de Poisson.
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Constructeur de la classe Poisson.

        Args:
            data (list, optional): Liste de données pour estimer la distribution. Par défaut, None.
            lambtha (float, optional): Nombre attendu d'occurrences dans une période donnée. Par défaut, 1.0.

        Raises:
            ValueError: Si lambtha n'est pas une valeur positive ou si les données ne contiennent pas assez de valeurs.
            TypeError: Si les données ne sont pas une liste.
        """
        if data is None:
            # Si les données ne sont pas fournies, utiliser lambtha
            if not isinstance(lambtha, (int, float)) or lambtha <= 0:
                raise ValueError("lambtha doit être une valeur positive")
            self.lambtha = float(lambtha)
        else:
            # Si les données sont fournies, calculer lambtha
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calcule la fonction de masse de probabilité (PMF) pour une valeur k.

        Args:
            k (int): Nombre d'occurrences.

        Returns:
            float: Probabilité que k occurrences se produisent.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        from math import exp, factorial
        return (exp(-self.lambtha) * (self.lambtha ** k)) / factorial(k)

    def cdf(self, k):
        """
        Calcule la fonction de distribution cumulative (CDF) pour une valeur k.

        Args:
            k (int): Nombre d'occurrences.

        Returns:
            float: Probabilité que le nombre d'occurrences soit inférieur ou égal à k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        from math import exp, factorial
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += (exp(-self.lambtha) * (self.lambtha ** i)) / factorial(i)
        return cdf_value
