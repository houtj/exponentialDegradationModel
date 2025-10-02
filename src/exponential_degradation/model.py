"""
Exponential Degradation Model for Remaining Useful Life Prediction.

This module implements the exponential degradation model described in:
Gebraeel, N. (2006). "Sensory-Updated Residual Life Distributions for 
Components With Exponential Degradation Patterns." IEEE Transactions on 
Automation Science and Engineering, 3(4), 382-393.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, Union


class ExponentialDegradationModel:
    """
    Exponential degradation model for predicting remaining useful life.
    
    This class implements a probabilistic exponential degradation model that 
    uses Bayesian updating to predict the remaining useful life (RUL) of a 
    system based on degradation measurements over time.
    
    The model assumes an exponential degradation path of the form:
        y(t) = exp(theta + beta*t) + phi
    
    where:
        - theta: initial value parameter (log-scale)
        - beta: growth rate parameter
        - phi: offset parameter
        - t: time
    
    Parameters
    ----------
    threshold : float
        The failure threshold value. System fails when degradation 
        measurement reaches this value.
    theta : float, optional
        Initial value parameter of the exponential model. 
        Default is 1.
    theta_variance : float, optional
        Initial variance (uncertainty) of the theta parameter.
        Large values indicate high uncertainty. Default is 1e6.
    beta : float, optional
        Growth rate parameter of the exponential model.
        Default is 1.
    beta_variance : float, optional
        Initial variance of the beta parameter.
        Large values indicate high uncertainty. Default is 1e6.
    rho : float, optional
        Correlation coefficient between theta and beta parameters,
        must be between -1 and 1. Default is 0 (uncorrelated).
    phi : float, optional
        Offset parameter of the exponential model. Default is -1.
        
    Attributes
    ----------
    threshold : float
        The failure threshold value.
    theta : float
        Current estimate of the initial value parameter.
    theta_p : float
        Log-transformed theta parameter.
    theta_variance : float
        Current variance of theta estimate.
    beta : float
        Current estimate of the growth rate parameter.
    beta_variance : float
        Current variance of beta estimate.
    rho : float
        Current correlation between theta and beta.
    phi : float
        Offset parameter.
    noise_variance : float
        Variance of measurement noise.
    ti : float
        Current time (last observation time).
        
    Examples
    --------
    >>> import numpy as np
    >>> from exponential_degradation import ExponentialDegradationModel
    >>> 
    >>> # Create model with failure threshold
    >>> model = ExponentialDegradationModel(threshold=10.0)
    >>> 
    >>> # Update with observations
    >>> times = np.array([1, 2, 3, 4, 5])
    >>> measurements = np.array([1.5, 2.1, 2.9, 4.2, 5.8])
    >>> model.fit(measurements, times)
    >>> 
    >>> # Predict remaining useful life
    >>> rul_result = model.predict_rul()
    >>> print(f"Predicted RUL: {rul_result['RUL']:.2f}")
    >>> print(f"95% CI: [{rul_result['CI'][0]:.2f}, {rul_result['CI'][1]:.2f}]")
    
    Notes
    -----
    The model uses Bayesian updating to refine parameter estimates as new 
    measurements become available. Initial parameter variances should be set 
    to large values when prior knowledge is weak, allowing the data to drive 
    the parameter estimates.
    
    References
    ----------
    .. [1] Gebraeel, N. (2006). "Sensory-Updated Residual Life Distributions 
           for Components With Exponential Degradation Patterns." IEEE 
           Transactions on Automation Science and Engineering, 3(4), 382-393.
    """
    
    def __init__(
        self,
        threshold: float,
        theta: float = 1.0,
        theta_variance: float = 1e6,
        beta: float = 1.0,
        beta_variance: float = 1e6,
        rho: float = 0.0,
        phi: float = -1.0,
    ):
        """Initialize the exponential degradation model."""
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a positive finite number")
        if not -1 <= rho <= 1:
            raise ValueError("rho must be between -1 and 1")
        if theta_variance <= 0 or beta_variance <= 0:
            raise ValueError("variances must be positive")
            
        self.threshold = threshold
        self.theta = theta
        self.theta_variance = theta_variance
        self.beta = beta
        self.beta_variance = beta_variance
        self.rho = rho
        self.phi = phi
        
        # Compute noise variance based on threshold
        self.noise_variance = (0.1 * threshold / (threshold + 1)) ** 2
        
        # Log-transformed theta parameter
        self.theta_p = np.log(self.theta - self.noise_variance**2 / 2)
        
        # Current time (last observation)
        self.ti = 0.0

    def update(
        self, 
        measurement: float, 
        time: float
    ) -> None:
        """
        Update model parameters based on a single new observation.
        
        This method performs Bayesian updating of the model parameters 
        (theta, beta, and their covariance) based on a single new degradation 
        measurement. The update equations are derived from the Kalman filter 
        framework for the exponential degradation model.
        
        Parameters
        ----------
        measurement : float
            Degradation measurement at the given time point.
            Must be positive and less than the failure threshold.
        time : float
            Time point at which the measurement was taken.
            Must be non-negative.
            
        Returns
        -------
        None
            Updates the model parameters in place.
            
        Raises
        ------
        ValueError
            If measurement or time are invalid (negative, non-finite, or 
            measurement exceeds threshold).
            
        See Also
        --------
        fit : Update parameters with multiple observations at once.
        
        Notes
        -----
        The update equations implement the posterior parameter distributions
        from equations (5)-(9) in Gebraeel (2006). The method updates:
        - theta_p (log-transformed initial value)
        - beta (growth rate)
        - theta_variance (uncertainty in theta)
        - beta_variance (uncertainty in beta)
        - rho (correlation between theta and beta)
        
        Examples
        --------
        >>> model = ExponentialDegradationModel(threshold=10.0)
        >>> model.update(measurement=2.5, time=1.0)
        >>> model.update(measurement=3.8, time=2.0)
        >>> print(f"Updated theta: {model.theta:.3f}")
        >>> print(f"Updated beta: {model.beta:.3f}")
        """
        if not np.isfinite(measurement) or measurement < 0:
            raise ValueError("measurement must be non-negative and finite")
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be non-negative and finite")
        if measurement >= self.threshold:
            raise ValueError("measurement must be less than threshold")
            
        self.ti = time
        
        # Current parameter values
        mu_0_p = self.theta_p
        mu_1 = self.beta
        sigma = self.noise_variance
        sigma_0 = self.theta_variance
        sigma_1 = self.beta_variance
        rho_0 = self.rho
        
        # Measurement value
        Li = measurement
        
        # Intermediate calculations for update equations
        X = (1 - rho_0**2) * sigma_0**2 + sigma**2
        Y = (1 - rho_0**2) * time**2 * sigma_1**2 + sigma**2
        M = (1 - rho_0**2) * time * sigma_0 * sigma_1 - rho_0 * sigma**2
        
        # Update theta_p (log-transformed theta)
        theta_p_term_1 = mu_0_p * sigma**2 * sigma_1 * (Y + M * rho_0)
        theta_p_term_2 = mu_1 * sigma**2 * sigma_0 * (Y * rho_0 + M)
        theta_p_term_3 = (1 - rho_0**2) * Li * sigma_0 * sigma_1 * (
            Y * sigma_0 - M * time * sigma_1
        )
        theta_p_updated = (
            (theta_p_term_1 - theta_p_term_2 + theta_p_term_3) /
            (sigma_1 * (X * Y - M**2))
        )
        
        # Update theta (exponential transformation)
        theta_updated = np.exp(theta_p_updated) + sigma**2 / 2
        
        # Update beta
        beta_term_1 = mu_1 * sigma**2 * sigma_0 * (X + M * rho_0)
        beta_term_2 = mu_0_p * sigma**2 * sigma_1 * (X * rho_0 + M)
        beta_term_3 = (1 - rho_0**2) * Li * sigma_0 * sigma_1 * (
            X * time * sigma_1 - M * sigma_0
        )
        beta_updated = (
            (beta_term_1 - beta_term_2 + beta_term_3) /
            (sigma_0 * (X * Y - M**2))
        )
        
        # Update theta variance
        theta_variance_numerator = (
            ((1 - rho_0**2) * time**2 * sigma_1**2 + sigma**2) *
            (1 - rho_0**2) * sigma**2 * sigma_0**2
        )
        theta_variance_denominator = (
            ((1 - rho_0**2) * sigma_0**2 + sigma**2) *
            ((1 - rho_0**2) * time**2 * sigma_1**2 + sigma**2) -
            ((1 - rho_0**2) * time * sigma_0 * sigma_1 - rho_0 * sigma**2)**2
        )
        theta_variance_updated = np.sqrt(
            theta_variance_numerator / theta_variance_denominator
        )
        
        # Update beta variance
        beta_variance_numerator = (
            ((1 - rho_0**2) * sigma_0**2 + sigma**2) *
            (1 - rho_0**2) * sigma**2 * sigma_1**2
        )
        beta_variance_denominator = theta_variance_denominator  # Same as theta
        beta_variance_updated = np.sqrt(
            beta_variance_numerator / beta_variance_denominator
        )
        
        # Update correlation coefficient
        rho_numerator = (1 - rho_0**2) * time * sigma_0 * sigma_1 - rho_0 * sigma**2
        rho_denominator = np.sqrt(
            ((1 - rho_0**2) * sigma_0**2 + sigma**2) *
            ((1 - rho_0**2) * time**2 * sigma_1**2 + sigma**2)
        )
        rho_updated = -(rho_numerator / rho_denominator)
        
        # Store updated parameters
        self.theta_p = theta_p_updated
        self.beta = beta_updated
        self.theta_variance = theta_variance_updated
        self.beta_variance = beta_variance_updated
        self.rho = rho_updated
        self.theta = theta_updated

    def fit(
        self,
        measurements: np.ndarray,
        times: np.ndarray
    ) -> None:
        """
        Update model parameters based on multiple observations.
        
        This method performs Bayesian updating of the model parameters using
        multiple degradation measurements simultaneously. It is more efficient
        than calling update() multiple times sequentially.
        
        Parameters
        ----------
        measurements : array-like of shape (n_samples,)
            Array of degradation measurements. All values must be positive 
            and less than the failure threshold.
        times : array-like of shape (n_samples,)
            Array of time points corresponding to each measurement.
            All values must be non-negative.
            
        Returns
        -------
        None
            Updates the model parameters in place.
            
        Raises
        ------
        ValueError
            If measurements and times have different lengths, or if any
            values are invalid.
            
        See Also
        --------
        update : Update parameters with a single observation.
        
        Notes
        -----
        The fit equations extend the single-update case to handle multiple
        observations by summing over all measurements. This is equivalent to
        processing all measurements simultaneously in a batch update.
        
        Examples
        --------
        >>> import numpy as np
        >>> model = ExponentialDegradationModel(threshold=10.0)
        >>> times = np.array([1, 2, 3, 4, 5])
        >>> measurements = np.array([1.5, 2.1, 2.9, 4.2, 5.8])
        >>> model.fit(measurements, times)
        >>> print(f"Fitted theta: {model.theta:.3f}")
        >>> print(f"Fitted beta: {model.beta:.3f}")
        """
        measurements = np.asarray(measurements)
        times = np.asarray(times)
        
        if measurements.shape != times.shape:
            raise ValueError("measurements and times must have the same shape")
        if len(measurements.shape) != 1:
            raise ValueError("measurements and times must be 1-dimensional")
        if not np.all(np.isfinite(measurements)) or np.any(measurements < 0):
            raise ValueError("measurements must be non-negative and finite")
        if not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must be non-negative and finite")
        if np.any(measurements >= self.threshold):
            raise ValueError("all measurements must be less than threshold")
            
        self.ti = times[-1]
        
        # Current parameter values
        mu_0_p = self.theta_p
        mu_1 = self.beta
        sigma = self.noise_variance
        sigma_0 = self.theta_variance
        sigma_1 = self.beta_variance
        rho_0 = self.rho
        
        L = measurements
        T = times
        k = len(measurements)
        
        # Intermediate calculations for batch update equations
        X = k * (1 - rho_0**2) * sigma_0**2 + sigma**2
        Y = (1 - rho_0**2) * sigma_1**2 * np.sum(T**2) + sigma**2
        M = (1 - rho_0**2) * sigma_0 * sigma_1 * np.sum(T) - rho_0 * sigma**2
        
        # Update theta_p
        theta_term_1 = mu_0_p * sigma**2 * sigma_1 * (Y + M * rho_0)
        theta_term_2 = mu_1 * sigma**2 * sigma_0 * (Y * rho_0 + M)
        theta_term_3 = (1 - rho_0**2) * sigma_0 * sigma_1 * (
            Y * sigma_0 * np.sum(L) - M * sigma_1 * np.sum(L * T)
        )
        theta_updated = (
            (theta_term_1 - theta_term_2 + theta_term_3) /
            (sigma_1 * (X * Y - M**2))
        )
        
        # Update beta
        beta_term_1 = mu_1 * sigma**2 * sigma_0 * (X + M * rho_0)
        beta_term_2 = mu_0_p * sigma**2 * sigma_1 * (X * rho_0 + M)
        beta_term_3 = (1 - rho_0**2) * sigma_0 * sigma_1 * (
            X * sigma_1 * np.sum(L * T) - M * sigma_0 * np.sum(L)
        )
        beta_updated = (
            (beta_term_1 - beta_term_2 + beta_term_3) /
            (sigma_0 * (X * Y - M**2))
        )
        
        # Update theta variance
        theta_variance_numerator = (
            ((1 - rho_0**2) * sigma_1**2 * np.sum(T**2) + sigma**2) *
            (1 - rho_0**2) * sigma**2 * sigma_0**2
        )
        theta_variance_denominator = (
            (k * (1 - rho_0**2) * sigma_0**2 + sigma**2) *
            ((1 - rho_0**2) * sigma_1**2 * np.sum(T**2) + sigma**2) -
            ((1 - rho_0**2) * sigma_0 * sigma_1 * np.sum(T) - rho_0 * sigma**2)**2
        )
        theta_variance_updated = np.sqrt(
            theta_variance_numerator / theta_variance_denominator
        )
        
        # Update beta variance
        beta_variance_numerator = (
            (k * (1 - rho_0**2) * sigma_0**2 + sigma**2) *
            (1 - rho_0**2) * sigma**2 * sigma_1**2
        )
        beta_variance_denominator = theta_variance_denominator  # Same as theta
        beta_variance_updated = np.sqrt(
            beta_variance_numerator / beta_variance_denominator
        )
        
        # Update correlation coefficient
        rho_numerator = (
            (1 - rho_0**2) * sigma_0 * sigma_1 * np.sum(T) - rho_0 * sigma**2
        )
        rho_denominator = np.sqrt(
            (k * (1 - rho_0**2) * sigma_0**2 + sigma**2) *
            ((1 - rho_0**2) * sigma_1**2 * np.sum(T**2) + sigma**2)
        )
        rho_updated = -(rho_numerator / rho_denominator)
        
        # Store updated parameters
        self.theta_p = theta_updated
        self.beta = beta_updated
        self.theta_variance = theta_variance_updated
        self.beta_variance = beta_variance_updated
        self.rho = rho_updated
        self.theta = theta_updated

    def predict(self) -> float:
        """
        Predict the mean remaining useful life (simple point estimate).
        
        This method provides a simple point estimate of the remaining useful
        life based on the current parameter estimates. For more comprehensive
        predictions including confidence intervals and probability distributions,
        use predict_rul() instead.
        
        Returns
        -------
        float
            Mean remaining useful life (RUL) estimate in the same time units
            as the input measurements.
            
        See Also
        --------
        predict_rul : Comprehensive RUL prediction with confidence intervals
                      and probability distributions.
        
        Notes
        -----
        The RUL is computed as:
            RUL = (ln(threshold - phi) - theta_p) / beta - t_current
        
        where t_current is the time of the last observation.
        
        Examples
        --------
        >>> model = ExponentialDegradationModel(threshold=10.0)
        >>> model.fit(measurements, times)
        >>> rul = model.predict()
        >>> print(f"Predicted RUL: {rul:.2f}")
        """
        end_of_life = (
            (np.log(self.threshold - self.phi) - self.theta_p) / self.beta
        )
        return end_of_life - self.ti

    def predict_rul(
        self,
        confidence_level: float = 0.95,
        num_samples: int = 1000
    ) -> Dict[str, Union[float, Tuple[float, float], np.ndarray, str]]:
        """
        Predict remaining useful life with confidence intervals and PDF.
        
        This method provides comprehensive RUL predictions including:
        - Point estimate (median)
        - Confidence intervals
        - Probability density function (PDF)
        - Cumulative distribution function (CDF)
        
        The method implements the truncated normal distribution approach from
        Gebraeel (2006), equations (10)-(12).
        
        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the confidence interval, between 0 and 1.
            Default is 0.95 (95% confidence interval).
        num_samples : int, optional
            Number of points to evaluate for the PDF and CDF.
            Default is 1000. More points provide smoother curves but
            increase computation time.
            
        Returns
        -------
        dict
            Dictionary containing the following keys:
            
            - 'RUL' : float
                Median remaining useful life (recommended point estimate).
                The median is preferred over the mean for skewed distributions.
            - 'mean' : float
                Mean of the truncated RUL distribution.
            - 'std' : float
                Standard deviation of the truncated RUL distribution.
            - 'CI' : tuple of (float, float)
                Confidence interval (lower_bound, upper_bound).
            - 'pdf_time' : ndarray
                Time points for PDF evaluation.
            - 'pdf_values' : ndarray
                Probability density values at pdf_time.
            - 'cdf_values' : ndarray
                Cumulative distribution values at pdf_time.
            - 'mu_untruncated' : float
                Mean of the untruncated normal distribution (before
                truncation at zero).
            - 'sigma_untruncated' : float
                Standard deviation of the untruncated normal distribution.
            - 'warning' : str, optional
                Warning message if numerical issues were detected.
                
        Raises
        ------
        ValueError
            If confidence_level is not between 0 and 1, or if num_samples
            is not positive.
            
        See Also
        --------
        predict : Simple point estimate of RUL.
        
        Notes
        -----
        The RUL distribution is based on the truncated normal distribution
        derived from the uncertainty in the model parameters. The distribution
        is truncated at zero to ensure only positive RUL values.
        
        The variance calculation accounts for the correlation between theta
        and beta parameters using the formula:
            Var(RUL) = (1/beta^2) * [Var(theta) + (mu_L*beta)^2 * Var(beta)
                       - 2*mu_L*beta*Cov(theta,beta)]
        
        References
        ----------
        .. [1] Gebraeel, N. (2006). "Sensory-Updated Residual Life 
               Distributions for Components With Exponential Degradation 
               Patterns." IEEE Transactions on Automation Science and 
               Engineering, 3(4), 382-393.
        
        Examples
        --------
        >>> model = ExponentialDegradationModel(threshold=10.0)
        >>> model.fit(measurements, times)
        >>> result = model.predict_rul(confidence_level=0.95)
        >>> print(f"Predicted RUL: {result['RUL']:.2f}")
        >>> print(f"Mean RUL: {result['mean']:.2f}")
        >>> print(f"95% CI: [{result['CI'][0]:.2f}, {result['CI'][1]:.2f}]")
        >>> 
        >>> # Plot the PDF
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result['pdf_time'], result['pdf_values'])
        >>> plt.xlabel('Time')
        >>> plt.ylabel('Probability Density')
        >>> plt.title('RUL Probability Distribution')
        >>> plt.show()
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        # Current time
        current_time = self.ti
        
        # Compute mean of the end-of-life distribution
        # Based on equations (8)-(9) from Gebraeel (2006)
        ln_threshold = np.log(self.threshold - self.phi)
        mu_L = (ln_threshold - self.theta_p) / self.beta
        
        # Variance calculation using propagation of uncertainty
        # For L = (ln(D-phi) - theta) / beta with correlated theta and beta
        # Var(L) = (1/beta^2) * [Var(theta) + (mu_L*beta)^2*Var(beta)
        #          - 2*mu_L*beta*Cov(theta,beta)]
        cov_theta_beta = self.rho * self.theta_variance * self.beta_variance
        
        var_L = (1.0 / (self.beta**2)) * (
            self.theta_variance**2 +
            (mu_L * self.beta)**2 * self.beta_variance**2 -
            2.0 * mu_L * self.beta * cov_theta_beta
        )
        
        # Ensure variance is positive
        var_L = max(var_L, 1e-10)
        sigma_L = np.sqrt(var_L)
        
        # Mean and std of remaining life R = L - t
        mu_R = mu_L - current_time
        sigma_R = sigma_L
        
        # Check for numerical validity
        if not np.isfinite(mu_R) or not np.isfinite(sigma_R) or sigma_R <= 0:
            # Return safe default values with warning
            return {
                'RUL': 0.0,
                'mean': 0.0,
                'CI': (0.0, 0.0),
                'pdf_time': np.array([0.0]),
                'pdf_values': np.array([0.0]),
                'cdf_values': np.array([1.0]),
                'std': 0.0,
                'mu_untruncated': mu_R,
                'sigma_untruncated': sigma_R,
                'warning': 'Numerical instability detected in RUL calculation'
            }
        
        # Compute truncated distribution parameters
        # The distribution is truncated at R > 0 (equation 11)
        alpha = -mu_R / sigma_R  # Standardized truncation point
        
        # Truncation probability (probability of negative RUL before truncation)
        Z_alpha = stats.norm.cdf(alpha)
        
        # Compute truncated normal statistics
        if Z_alpha < 0.9999:  # Avoid numerical issues
            lambda_alpha = stats.norm.pdf(alpha) / (1 - Z_alpha)
            mean_truncated = mu_R + sigma_R * lambda_alpha
            
            # Variance of truncated normal
            delta_alpha = lambda_alpha * (lambda_alpha - alpha)
            var_truncated = sigma_R**2 * (1 - delta_alpha)
            var_truncated = max(var_truncated, 1e-10)
            std_truncated = np.sqrt(var_truncated)
        else:
            # If truncation probability is too high, use exponential approximation
            mean_truncated = max(sigma_R / 10.0, 0.001)
            std_truncated = mean_truncated
        
        # Compute median of truncated distribution
        try:
            median_rul = self._truncated_normal_ppf(0.5, mu_R, sigma_R, 0, np.inf)
            if not np.isfinite(median_rul):
                median_rul = max(mean_truncated, 0.001)
        except Exception:
            median_rul = max(mean_truncated, 0.001)
        
        # Compute confidence intervals
        alpha_ci = (1 - confidence_level) / 2
        try:
            ci_lower = self._truncated_normal_ppf(alpha_ci, mu_R, sigma_R, 0, np.inf)
            ci_upper = self._truncated_normal_ppf(
                1 - alpha_ci, mu_R, sigma_R, 0, np.inf
            )
            if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
                ci_lower = max(mean_truncated - 2 * std_truncated, 0.0)
                ci_upper = mean_truncated + 2 * std_truncated
        except Exception:
            ci_lower = max(mean_truncated - 2 * std_truncated, 0.0)
            ci_upper = mean_truncated + 2 * std_truncated
        
        # Generate PDF and CDF values
        max_time = max(ci_upper * 1.5, mean_truncated + 4 * std_truncated, 1.0)
        if not np.isfinite(max_time):
            max_time = mean_truncated + 4 * std_truncated
        pdf_time = np.linspace(0, max_time, num_samples)
        
        # Compute PDF values (equation 12)
        pdf_values = self._truncated_normal_pdf(pdf_time, mu_R, sigma_R, 0, np.inf)
        
        # Compute CDF values (equation 11)
        cdf_values = self._truncated_normal_cdf(pdf_time, mu_R, sigma_R, 0, np.inf)
        
        return {
            'RUL': median_rul,
            'mean': mean_truncated,
            'CI': (ci_lower, ci_upper),
            'pdf_time': pdf_time,
            'pdf_values': pdf_values,
            'cdf_values': cdf_values,
            'std': std_truncated,
            'mu_untruncated': mu_R,
            'sigma_untruncated': sigma_R
        }

    def _truncated_normal_pdf(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
        a: float,
        b: float
    ) -> np.ndarray:
        """
        Compute PDF of truncated normal distribution.
        
        Implements equation (12) from Gebraeel (2006).
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF.
        mu : float
            Mean of the untruncated normal distribution.
        sigma : float
            Standard deviation of the untruncated normal distribution.
        a : float
            Lower truncation point.
        b : float
            Upper truncation point (use np.inf for one-sided truncation).
            
        Returns
        -------
        ndarray
            PDF values at x. Values outside [a, b] are zero.
            
        Notes
        -----
        The PDF of a truncated normal distribution is:
            f(x) = phi((x-mu)/sigma) / (sigma * Z)  for a <= x <= b
            f(x) = 0                                 otherwise
        where phi is the standard normal PDF and Z is the normalization constant.
        """
        x = np.asarray(x)
        
        # Standardize
        z = (x - mu) / sigma
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        
        # Normalization constant (probability mass in truncated region)
        Z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        
        # Avoid division by zero
        if Z <= 0:
            return np.zeros_like(x)
        
        # PDF of truncated normal
        pdf = np.where(
            (x >= a) & (x <= b),
            stats.norm.pdf(z) / (sigma * Z),
            0.0
        )
        
        return pdf

    def _truncated_normal_cdf(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
        a: float,
        b: float
    ) -> np.ndarray:
        """
        Compute CDF of truncated normal distribution.
        
        Implements equation (11) from Gebraeel (2006).
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate the CDF.
        mu : float
            Mean of the untruncated normal distribution.
        sigma : float
            Standard deviation of the untruncated normal distribution.
        a : float
            Lower truncation point.
        b : float
            Upper truncation point (use np.inf for one-sided truncation).
            
        Returns
        -------
        ndarray
            CDF values at x. Values are 0 for x < a and 1 for x > b.
            
        Notes
        -----
        The CDF of a truncated normal distribution is:
            F(x) = 0                                    for x < a
            F(x) = [Phi((x-mu)/sigma) - Phi(alpha)] / Z for a <= x <= b
            F(x) = 1                                    for x > b
        where Phi is the standard normal CDF and Z is the normalization constant.
        """
        x = np.asarray(x)
        
        # Standardize
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        
        # Normalization constant
        Z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        
        # Avoid division by zero
        if Z <= 0:
            return np.ones_like(x)
        
        # CDF of truncated normal
        cdf = np.zeros_like(x)
        cdf = np.where(x < a, 0.0, cdf)
        cdf = np.where(x > b, 1.0, cdf)
        
        mask = (x >= a) & (x <= b)
        if np.any(mask):
            z = (x[mask] - mu) / sigma
            cdf[mask] = (stats.norm.cdf(z) - stats.norm.cdf(alpha)) / Z
        
        return cdf

    def _truncated_normal_ppf(
        self,
        p: float,
        mu: float,
        sigma: float,
        a: float,
        b: float
    ) -> float:
        """
        Compute inverse CDF (quantile function) of truncated normal distribution.
        
        Parameters
        ----------
        p : float
            Probability (between 0 and 1).
        mu : float
            Mean of the untruncated normal distribution.
        sigma : float
            Standard deviation of the untruncated normal distribution.
        a : float
            Lower truncation point.
        b : float
            Upper truncation point (use np.inf for one-sided truncation).
            
        Returns
        -------
        float
            Quantile value at probability p.
            
        Notes
        -----
        The quantile function inverts the CDF:
            x = F^(-1)(p) such that P(X <= x) = p
        for the truncated normal distribution.
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
            
        # Standardize truncation points
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        
        # CDF values at truncation points
        Phi_alpha = stats.norm.cdf(alpha)
        Phi_beta = stats.norm.cdf(beta)
        
        # Transform p to the scale of the standard normal
        q = Phi_alpha + p * (Phi_beta - Phi_alpha)
        
        # Inverse CDF of standard normal
        z = stats.norm.ppf(q)
        
        # Transform back to original scale
        return mu + sigma * z

