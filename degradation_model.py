#%%
import numpy as np
from scipy import stats
from typing import Tuple, Optional

class exponentialDegradationModel():
    """Exponential Degradation Model for predicting remaining useful life.
    This class implements a probabilistic exponential degradation model that can be used
    to predict the remaining useful life of a system based on degradation measurements.
    Parameters
    ----------
    threshold : float
        The failure threshold value for the system
    theta : float, optional (default=1)
        Initial value parameter of the exponential model
    theta_variance : float, optional (default=1e6) 
        Initial variance of theta parameter
    beta : float, optional (default=1)
        Growth rate parameter of the exponential model
    beta_variance : float, optional (default=1e6)
        Initial variance of beta parameter
    rho : float, optional (default=0)
        Correlation coefficient between theta and beta
    phi : float, optional (default=-1)
        Offset parameter of the exponential model
    Methods
    -------
    update(Li_original, ti)
        Updates model parameters based on a single new observation
    fit(L_original, T) 
        Updates model parameters based on multiple observations
    predict()
        Predicts the remaining useful life
    plot(L, T, remaining_life)
        Visualizes the degradation path and predictions
    Notes
    -----
    The model assumes an exponential degradation path of the form:
    y = exp(theta + beta*t) + phi
    where:
    - theta is the initial value parameter
    - beta is the growth rate parameter  
    - phi is an offset parameter
    - t is time
    The model uses Bayesian updating to refine parameter estimates as new 
    measurements become available.
    """
    def __init__(self, 
                 threshold,
                 theta=1,
                 theta_variance=1e6,
                 beta=1,
                 beta_variance=1e6,
                 rho=0,
                 phi=-1,
                 ):
        self.threshold = threshold
        self.theta = theta
        self.theta_variance=theta_variance
        self.beta=beta
        self.beta_variance= beta_variance
        self.rho=rho
        self.phi = phi
        self.noise_variance=(0.1*threshold/(threshold + 1))**2
        self.theta_p = np.log(self.theta-self.noise_variance**2/2)

    def update(self, Li_original, ti):
        self.ti = ti
        mu_0_p = self.theta_p
        mu_1 = self.beta
        sigma = self.noise_variance
        sigma_0 = self.theta_variance
        sigma_1 = self.beta_variance
        rho_0 = self.rho
        # Li = np.log(Li_original)
        Li = Li_original

        X = (1-rho_0**2)*sigma_0**2 + sigma**2
        Y = (1-rho_0**2)*ti**2*sigma_1**2+sigma**2
        M = (1-rho_0**2)*ti*sigma_0*sigma_1-rho_0*sigma**2

        theta_p_term_1 = mu_0_p*sigma**2*sigma_1*(Y+M*rho_0)
        theta_p_term_2 = mu_1*sigma**2*sigma_0*(Y*rho_0+M)
        theta_p_term_3 = (1-rho_0**2)*Li*sigma_0*sigma_1*(Y*sigma_0-M*ti*sigma_1)
        theta_p_updated_top = theta_p_term_1-theta_p_term_2+theta_p_term_3
        theta_p_updated_bottom = sigma_1*(X*Y-M**2)
        theta_p_updated = theta_p_updated_top/theta_p_updated_bottom
        theta_updated = np.exp(theta_p_updated)+sigma**2/2


        beta_term_1 = mu_1*sigma**2*sigma_0*(X+M*rho_0)
        beta_term_2 = mu_0_p*sigma**2*sigma_1*(X*rho_0+M)
        beta_term_3 = (1-rho_0**2)*Li*sigma_0*sigma_1*(X*ti*sigma_1-M*sigma_0)
        beta_updated_top = beta_term_1-beta_term_2+beta_term_3
        beta_updated_bottom = sigma_0*(X*Y-M**2)
        beta_updated = beta_updated_top / beta_updated_bottom

        theta_variance_updated_top = ((1-rho_0**2)*ti**2*sigma_1**2+sigma**2)*(1-rho_0**2)*sigma**2*sigma_0**2
        theta_variance_term_1 = ((1-rho_0**2)*sigma_0**2+sigma**2)*((1-rho_0**2)*ti**2*sigma_1**2+sigma**2)
        theta_variance_term_2 = ((1-rho_0**2)*ti*sigma_0*sigma_1-rho_0*sigma**2)**2
        theta_variance_updated_bottom = theta_variance_term_1-theta_variance_term_2
        theta_variance_updated = np.sqrt(theta_variance_updated_top / theta_variance_updated_bottom)

        beta_variance_updated_top = ((1-rho_0**2)*sigma_0**2+sigma**2)*(1-rho_0**2)*sigma**2*sigma_1**2
        beta_variance_term_1 = ((1-rho_0**2)*sigma_0**2+sigma**2)*((1-rho_0**2)*ti**2*sigma_1**2+sigma**2)
        beta_variance_term_2 = ((1-rho_0**2)*ti*sigma_0*sigma_1-rho_0*sigma**2)**2
        beta_variance_updated_bottom = beta_variance_term_1-beta_variance_term_2
        beta_variance_updated = np.sqrt(beta_variance_updated_top / beta_variance_updated_bottom)

        rho_updated_top = (1-rho_0**2)*ti*sigma_0*sigma_1-rho_0*sigma**2
        rho_updated_bottom = np.sqrt( ((1-rho_0**2)*sigma_0**2+sigma**2)*((1-rho_0**2)*ti**2*sigma_1**2+sigma**2) )
        rho_updated = - (rho_updated_top/rho_updated_bottom)

        self.theta_p = theta_p_updated
        self.beta = beta_updated
        self.theta_variance = theta_variance_updated
        self.beta_variance = beta_variance_updated
        self.rho = rho_updated
        self.theta = theta_updated
    
    def fit(self, L_original, T):
        self.ti = T[-1]
        mu_0_p = self.theta_p
        mu_1 = self.beta
        sigma = self.noise_variance
        sigma_0 = self.theta_variance
        sigma_1 = self.beta_variance
        rho_0 = self.rho
        # L = np.log(L_original)
        L = L_original
        k = L.shape[0]

        X = k*(1-rho_0**2)*sigma_0**2 + sigma**2
        Y = (1-rho_0**2)*sigma_1**2*np.sum(T**2) + sigma**2
        M = (1-rho_0**2)*sigma_0*sigma_1*np.sum(T) - rho_0*sigma**2

        theta_term_1 = mu_0_p*sigma**2*sigma_1*(Y+M*rho_0)
        theta_term_2 = mu_1*sigma**2*sigma_0*(Y*rho_0+M)
        theta_term_3 = (1-rho_0**2)*sigma_0*sigma_1*(Y*sigma_0*np.sum(L) - M*sigma_1*np.sum(np.multiply(L,T)))
        theta_updated_top = theta_term_1 - theta_term_2 + theta_term_3
        theta_updated_bottom = sigma_1*(X*Y-M**2)
        theta_updated = theta_updated_top/theta_updated_bottom

        beta_term_1 = mu_1*sigma**2*sigma_0*(X+M*rho_0)
        beta_term_2 = mu_0_p*sigma**2*sigma_1*(X*rho_0+M)
        beta_term_3 = (1-rho_0**2)*sigma_0*sigma_1*(X*sigma_1*np.sum(np.multiply(L, T))-M*sigma_0*np.sum(L))
        beta_updated_top = beta_term_1 - beta_term_2 + beta_term_3
        beta_updated_bottom = sigma_0*(X*Y-M**2)
        beta_updated = beta_updated_top / beta_updated_bottom

        theta_variance_updated_top = ((1-rho_0**2)*sigma_1**2*np.sum(T**2)+sigma**2)*(1-rho_0**2)*sigma**2*sigma_0**2
        theta_variance_term_1 = (k*(1-rho_0**2)*sigma_0**2+sigma**2)*((1-rho_0**2)*sigma_1**2*np.sum(T**2)+sigma**2)
        theta_variance_term_2 = ((1-rho_0**2)*sigma_0*sigma_1*np.sum(T)-rho_0*sigma**2)**2
        theta_variance_updated_bottom = theta_variance_term_1 - theta_variance_term_2
        theta_variance_updated = np.sqrt(theta_variance_updated_top / theta_variance_updated_bottom)

        beta_variance_updated_top = (k*(1-rho_0**2)*sigma_0**2+sigma**2)*(1-rho_0**2)*sigma**2*sigma_1**2
        beta_variance_term_1 = (k*(1-rho_0**2)*sigma_0**2+sigma**2)*((1-rho_0**2)*sigma_1**2*np.sum(T**2)+sigma**2)
        beta_variance_term_2 = ((1-rho_0**2)*sigma_0*sigma_1*np.sum(T)-rho_0*sigma**2)**2
        beta_variance_updated_bottom = beta_variance_term_1 - beta_variance_term_2
        beta_variance_updated = np.sqrt(beta_variance_updated_top / beta_variance_updated_bottom)       
    
        rho_updated_top = (1-rho_0**2)*sigma_0*sigma_1*np.sum(T)-rho_0*sigma**2
        rho_updated_bottom = np.sqrt((k*(1-rho_0**2)*sigma_0**2+sigma**2)*((1-rho_0**2)*sigma_1**2*np.sum(T**2)+sigma**2))
        rho_updated = - (rho_updated_top/rho_updated_bottom)

        self.theta_p = theta_updated
        self.beta = beta_updated
        self.theta_variance = theta_variance_updated
        self.beta_variance = beta_variance_updated
        self.rho = rho_updated
        self.theta = theta_updated

    def predict(self) -> float:
        """Predict the mean remaining useful life.
        
        Returns
        -------
        float
            Mean remaining useful life (RUL) estimate
        """
        end_of_life = (np.log(self.threshold-self.phi)-self.theta_p)/self.beta
        return end_of_life-self.ti
    
    def predictRUL(self, confidence_level: float = 0.95, num_samples: int = 1000) -> dict:
        """Predict remaining useful life with confidence intervals and PDF.
        
        This method implements functionality similar to MATLAB's predictRUL function,
        computing the RUL estimate, confidence intervals, and probability density function
        based on the truncated normal distribution derived from the exponential degradation model.
        
        Parameters
        ----------
        confidence_level : float, optional (default=0.95)
            Confidence level for the confidence interval (e.g., 0.95 for 95% CI)
        num_samples : int, optional (default=1000)
            Number of points to evaluate for the PDF
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'RUL': float, median remaining useful life (recommended estimate)
            - 'mean': float, mean remaining useful life
            - 'CI': tuple of (lower, upper), confidence interval bounds
            - 'pdf_time': ndarray, time points for PDF evaluation
            - 'pdf_values': ndarray, probability density values at pdf_time
            - 'cdf_values': ndarray, cumulative distribution values at pdf_time
            
        Notes
        -----
        The RUL distribution is based on equations (10)-(12) from Gebraeel (2006):
        "Sensory-Updated Residual Life Distributions for Components With 
        Exponential Degradation Patterns"
        
        The distribution is truncated at 0 to ensure only positive RUL values.
        The median is recommended as the point estimate since the distribution
        is typically skewed.
        """
        # Current time
        current_time = self.ti
        
        # Compute mean and variance of the remaining life distribution
        # Based on equation (8) and (9) from the paper
        ln_threshold = np.log(self.threshold - self.phi)
        mu_L = (ln_threshold - self.theta_p) / self.beta
        
        # Variance calculation from the paper using propagation of uncertainty
        # For L = (ln(D-phi) - theta) / beta, where theta and beta are correlated
        # Var(L) = (1/beta^2) * [Var(theta) + (mu_L^2)*Var(beta) - 2*mu_L*Cov(theta,beta)]
        # where Cov(theta,beta) = rho * sigma_theta * sigma_beta
        
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
            # Return safe default values
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
        
        # Compute the truncated distribution parameters
        # The distribution is truncated at R > 0 (equation 11)
        alpha = -mu_R / sigma_R  # Standardized truncation point
        
        # Truncation probability (probability of negative RUL before truncation)
        Z_alpha = stats.norm.cdf(alpha)
        
        # For truncated normal distribution:
        # Mean of truncated normal
        if Z_alpha < 0.9999:  # Avoid numerical issues when almost all mass is below 0
            lambda_alpha = stats.norm.pdf(alpha) / (1 - Z_alpha)
            mean_truncated = mu_R + sigma_R * lambda_alpha
            
            # Variance of truncated normal
            delta_alpha = lambda_alpha * (lambda_alpha - alpha)
            var_truncated = sigma_R**2 * (1 - delta_alpha)
            var_truncated = max(var_truncated, 1e-10)  # Ensure positive
            std_truncated = np.sqrt(var_truncated)
        else:
            # If truncation probability is too high, most mass is below 0
            # Use exponential approximation for the positive tail
            mean_truncated = max(sigma_R / 10.0, 0.001)
            std_truncated = mean_truncated
            
        # Compute median of truncated distribution
        # For truncated normal, median can be computed by finding the 50th percentile
        try:
            median_rul = self._truncated_normal_ppf(0.5, mu_R, sigma_R, 0, np.inf)
            if not np.isfinite(median_rul):
                median_rul = max(mean_truncated, 0.001)
        except:
            median_rul = max(mean_truncated, 0.001)
        
        # Compute confidence intervals
        alpha_ci = (1 - confidence_level) / 2
        try:
            ci_lower = self._truncated_normal_ppf(alpha_ci, mu_R, sigma_R, 0, np.inf)
            ci_upper = self._truncated_normal_ppf(1 - alpha_ci, mu_R, sigma_R, 0, np.inf)
            if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
                ci_lower = max(mean_truncated - 2*std_truncated, 0.0)
                ci_upper = mean_truncated + 2*std_truncated
        except:
            ci_lower = max(mean_truncated - 2*std_truncated, 0.0)
            ci_upper = mean_truncated + 2*std_truncated
        
        # Generate PDF and CDF values
        # Create time points from 0 to upper CI + buffer
        max_time = max(ci_upper * 1.5, mean_truncated + 4*std_truncated, 1.0)
        if not np.isfinite(max_time):
            max_time = mean_truncated + 4*std_truncated
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
    
    def _truncated_normal_pdf(self, x: np.ndarray, mu: float, sigma: float, 
                             a: float, b: float) -> np.ndarray:
        """Compute PDF of truncated normal distribution.
        
        Implements equation (12) from Gebraeel (2006).
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF
        mu : float
            Mean of the untruncated normal distribution
        sigma : float
            Standard deviation of the untruncated normal distribution
        a : float
            Lower truncation point
        b : float
            Upper truncation point
            
        Returns
        -------
        ndarray
            PDF values at x
        """
        x = np.asarray(x)
        
        # Standardize
        z = (x - mu) / sigma
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        
        # Normalization constant (probability mass in truncated region)
        Z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        
        # PDF of truncated normal
        pdf = np.where((x >= a) & (x <= b), 
                      stats.norm.pdf(z) / (sigma * Z),
                      0.0)
        
        return pdf
    
    def _truncated_normal_cdf(self, x: np.ndarray, mu: float, sigma: float,
                             a: float, b: float) -> np.ndarray:
        """Compute CDF of truncated normal distribution.
        
        Implements equation (11) from Gebraeel (2006).
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate the CDF
        mu : float
            Mean of the untruncated normal distribution
        sigma : float
            Standard deviation of the untruncated normal distribution
        a : float
            Lower truncation point
        b : float
            Upper truncation point
            
        Returns
        -------
        ndarray
            CDF values at x
        """
        x = np.asarray(x)
        
        # Standardize
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        
        # Normalization constant
        Z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        
        # CDF of truncated normal
        cdf = np.zeros_like(x)
        cdf = np.where(x < a, 0.0, cdf)
        cdf = np.where(x > b, 1.0, cdf)
        
        mask = (x >= a) & (x <= b)
        z = (x[mask] - mu) / sigma
        cdf[mask] = (stats.norm.cdf(z) - stats.norm.cdf(alpha)) / Z
        
        return cdf
    
    def _truncated_normal_ppf(self, p: float, mu: float, sigma: float,
                             a: float, b: float) -> float:
        """Compute inverse CDF (quantile function) of truncated normal distribution.
        
        Parameters
        ----------
        p : float
            Probability (between 0 and 1)
        mu : float
            Mean of the untruncated normal distribution
        sigma : float
            Standard deviation of the untruncated normal distribution
        a : float
            Lower truncation point
        b : float
            Upper truncation point
            
        Returns
        -------
        float
            Quantile value at probability p
        """
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