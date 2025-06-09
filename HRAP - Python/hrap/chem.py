# Author: Thomas A. Scott, https://www.scott-aero.com/

# HP Gibb's free energy minimization (specified input enthalpy and pressure)
# Note: RPA, CEA, PROPEP all use the same algorithm
# Why make this? Because pypropep is broken and abandoned, pycea requires uses to install Fortran, and Cantera's equilibrium solver has been broken for over a decade. That is, there are no suitable Python alternatives.

# Condensed species are mostly implemented but disabled as the procedure of solving for gas only before adding condensed to solve coupled proved necessary

# PORTING AND UNIT TESTS ARE WIP!



# Terminlogy explanations:
# Mixture - reaction products

# Unit explanations:
# s - specific entropy
# S - molar entropy
# h - specific enthalpy
# H - molar enthalpy
# g - Gibbs energy per kg of mixture
# mu_j - chemical potential per kmol of species j

# Extension explanations:
# _0 - fixed/initial value
# ^0 - standard state value
# _D - dimensionless value

# Working equations:
# s = sum n_j*S_j
# h = sum n_j*H_j
# g = sum n_j*mu_j

import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Dict



Rhat = 8314 # J/(K*kmol), universal gas constant

# @partial(jax.tree_util.register_dataclass,
#     data_fields=[],
#     meta_fields=[])
@dataclass
class NASA9(object):
    T_min: float # K
    T_max: float # K
    # TODO: check consistent with curve
    dho: float # Formation enthalpy
    coeffs: list[float] # 9 coefficients
    
    # Nondimensionalized specific heat, Cp/Rhat
    def get_Cp_D(self, T):
        return self.coeffs[0]/(T*T) + self.coeffs[1]/T   + self.coeffs[2]      + \
               self.coeffs[3]*T     + self.coeffs[4]*T*T + self.coeffs[5]*T**2 + \
               self.coeffs[6]*T**4
    
    # Nondimensionalized specific enthalpy, H/(Rhat*T)
    def get_H_D(self, T):
         return -self.coeffs[0]/(T*T)    + self.coeffs[1]/T*np.log(T) + self.coeffs[2]          + \
                 self.coeffs[3]*T/2.0    + self.coeffs[4]*T*T/3.0     + self.coeffs[5]*T**3/4.0 + \
                 self.coeffs[6]*T**4/5.0 + self.coeffs[7]/T

    # Nondimensionalized specific entropy, S/Rhat
    def get_S_D(self, T):
        return -self.coeffs[0]/(2.0*T*T) - self.coeffs[1]/T       + self.coeffs[2]*np.log(T) + \
                self.coeffs[3]*T         + self.coeffs[4]*T*T/2.0 + self.coeffs[5]*T**3/3.0  + \
                self.coeffs[6]*T**4/4.0  + self.coeffs[8]

# @partial(jax.tree_util.register_dataclass,
#     data_fields=[],
#     meta_fields=[])
@dataclass
class ThermoSubstance(object):
    formula: str # Format should be similar to "(HCOOH)2-", all capitals
    comment: str # Such as data origin
    condensed: bool # Either condensed or gaseous
    is_product: bool # Always a reactant, not always a product
    composition: Dict[str, float] # two-letter code: relative moles i.e. chemical composition
    M: float # kg/kmol, molar mass
    providers: list[NASA9]
    T_min: float # Range where the substance is allowed to be used (may be a single point, may contain error margin past where it is defined)
    T_max: float

    def get_R(self): # J/(K*kg), specific gas constant
        return Rhat / self.M
    
    def get_prov(self, T):
        i = 0
        for j in range(1, len(self.providers)):
            if T > self.providers[j].T_min: i = j
        
        return self.providers[i]

    def get_Cp_D(self, T):
        return self.get_prov(T).get_Cp_D(T)
    
    def get_H_D(self, T):
        return self.get_prov(T).get_H_D(T)
        
    def get_S_D(self, T):
        return self.get_prov(T).get_S_D(T)

def make_basic_reactant(formula: str, composition: dict, M: float, T0: float, h0: float, condensed = True) -> ThermoSubstance:
    """
    kg/kmol
    K
    J/mol
    """
    return ThermoSubstance(formula, '', condensed, False, { k.upper(): v for k, v in composition.items() }, M, [NASA9(0.9*T0, 1.1*T0, h0, [0.0]*2+[h0/Rhat/T0]+[0.0]*5)], 0.9*T0, 1.1*T0)

@dataclass
class ChemSolver:
    substances: Dict[str, ThermoSubstance]

    # TODO: remove std from comments
    
    # Takes in table such as thermo.ipa from RPA or thermo.dat from cpropep. Must be text, not binary
    # Must end with END REACTANTS and contain END PRODUCTS
    # Curve fit data should be dimensionless (Cp/R, S/R, and H/R) with S and H supplied at 1bar
    # Curve fit for Cp can not have more than 7 elements and exponents (-2, -1, 0, 1, 2, 3, 4) are always assumped, with data ignored
    def load_propep(self, chem_path):
        # Reads from entries such as:
        """
        name (18 char), comment (all remaining on line)
        H2O               Hf:Cox,1989. Woolley,1987. TRC(10/88) tuv25.                  
        temperature range entry count (2 char, kelvin), source identifier (8x char), elements (5x [2x char element, 6x char quantity]]), phase (2char, 0 for gas 1 for condensed), molecular weight (13 char, kg/kmol), heat of formation (15 char, J/mol, or enthalpy if entry count = 0 [single temperature, condensed])
         2 g 8/89 H   2.00O   1.00    0.00    0.00    0.00 0   18.0152800    -241826.000
        temperature interval (2x 11 char), ncoeffs (1 char), exponents (8x 5 char), H^O(298.15)-H^O(0) (17 char, ?)
            200.000   1000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         9904.092
        coefficients (across 2 lines, up to 8x 16 char, may be blank to imply skip for formatting)
        -3.947960830D+04 5.755731020D+02 9.317826530D-01 7.222712860D-03-7.342557370D-06
         4.955043490D-09-1.336933246D-12                -3.303974310D+04 1.724205775D+01
        (repeated for other temperature intervals)
           1000.000   6000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         9904.092
         1.034972096D+06-2.412698562D+03 4.646110780D+00 2.291998307D-03-6.836830480D-07
         9.426468930D-11-4.822380530D-15                -1.384286509D+04-7.978148510D+00
        """
        with open(chem_path, 'r') as chem_file:
            substances = { }
            strip = ' '
            mode, in_reactants = 0, False
            def readline(chem_file):
                line = chem_file.readline().rstrip('\n')
                while len(line.strip(strip)) == 0 or line[0] == '!': # Skip comments and blank lines
                    line = chem_file.readline().rstrip('\n')
                return line
            line = readline(chem_file)
            while line:
                if line.startswith('thermo'):
                    chem_file.readline() # Skip line containing extraneous range data
                elif line.startswith('END PRODUCTS'):
                    in_reactants = True
                elif line.startswith('END REACTANTS'):
                    break # End of file
                else: # Chem entry
                    i = 0
                    
                    formula = line[i:i+18].strip(strip); i += 18
                    comment = ''
                    if len(line) > 18:
                        comment = line[i:].strip(strip)
                    
                    line = readline(chem_file); i = 0
                    fitPieces = int(line[i:i+2].strip(strip)); i += 2
                    i += 8 # Ignore source identifier
                    composition = { }
                    for j in range(5):
                        symbol = line[i:i+2].strip(strip).upper(); i += 2
                        quantity = float(line[i:i+6].strip(strip)); i += 6
                        if len(symbol) == 0:
                            continue
                        composition[symbol] = quantity
                    phase = int(line[i:i+2].strip(strip)); i += 2
                    condensed = phase != 0 # TODO: store phase?
                    M = float(line[i:i+13].strip(strip)); i += 13
                    # Comes as kJ/kmol so convert to J/kmol
                    DeltaHForm = 1000 * float(line[i:i+15].strip(strip))

                    line = readline(chem_file); i = 0
                    providers = []
                    if fitPieces == 0: # If single temp (only appropriate for condensed reactants)
                        if not in_reactants:
                            print(f'ERROR: Zero piece data is not permitted for product, {formula}, as entropy is needed to solve!')
                        
                        T = float(line[i:i+11].strip(strip))
                        # Valid at a single temperature but is specified as constant Cp
                        providers.append(NASA9(0.9*T, 1.1*T, DeltaHForm, [0.0]*2+[DeltaHForm/Rhat/T]+[0.0]*5))
                    else:
                        for j in range(fitPieces):
                            i = 0
                            T_min = float(line[i:i+11].strip(strip)); i += 11
                            T_max = float(line[i:i+11].strip(strip)); i += 11
                            ncoffs = int(line[i:i+1].strip(strip)); i += 1
                            if ncoffs > 8:
                                print(f'Fatal error: Thermochemical substance {formula} cannot have more than 8 coefficients (not counting the 9th entropy constant)!')
                                return
                            # TODO: warning if <8?
                            i += 8*5 # Ignore exponents (only specified for C_p and have to make consistent assumptions for others)
                            dho = float(line[i:i+17].strip(strip))

                            coeffs, m = [0.0]*9, 0
                            line = readline(chem_file); i = 0
                            for k in range(5):
                                coeff = line[i:i+16].strip(strip); i += 16
                                if len(coeff) == 0: continue
                                coeffs[m] = float(coeff.replace('D', 'E')); m += 1
                            
                            line = readline(chem_file); i = 0
                            for k in range(2):
                                coeff = line[i:i+16].strip(strip); i += 16
                                if len(coeff) == 0: continue
                                coeffs[m] = float(coeff.replace('D', 'E')); m += 1
                            i += 16 # center is always ignored
                            # Last two are for dimensionless molar enthalpy and entropy, respectively
                            for k in range(2):
                                coeff = line[i:i+16].strip(strip); i += 16
                                if len(coeff) == 0: continue
                                coeffs[7+k] = float(coeff.replace('D', 'E'))
                            providers.append(NASA9(T_min, T_max, DeltaHForm, coeffs))
                            # formula: str # Format should be similar to "(HCOOH)2-", all capitals
                            if j != fitPieces-1: line = readline(chem_file)
                    T_min, T_max = np.min([prov.T_min for prov in providers]), np.max([prov.T_max for prov in providers])
                    substances[formula] = ThermoSubstance(formula, comment, condensed, not in_reactants, composition, M, providers, T_min, T_max)
                
                line = readline(chem_file)
        return substances
    
    def __init__(self, chem_infos):
        self.substances = { }
        if not (isinstance(chem_infos, list) or isinstance(chem_infos, tuple)):
            chem_infos = [chem_infos]
        for chem_info in chem_infos:
            # TODO: check for duplicates!
            if isinstance(chem_info, str) or isinstance(chem_info, Path):
                new_substances = self.load_propep(chem_info)
                for k, v in new_substances.items():
                    if k in self.substances:
                        print('Warning chemical "{k}" was defined several times, only first occurance used'.format(k=k))
                    else:
                        self.substances[k] = v
            elif isinstance(chem_info, ThermoSubstance):
                if chem_info.formula in self.substances:
                    print('Warning chemical "{k}" was defined several times, only first occurance used'.format(k=chem_info.formula))
                else:
                    self.substances[chem_info.formula] = chem_info
            else:
                print('Error: invalid chem info type', type(chem_info))

    #void Supply(ThermochemicalSubstance *substance, m_frac, T, P); # mass fraction, T(K), P(Pa)
    # Result Solve(P, int iterations, bool returnComposition = false); # P(Pa)

    # TODO: Take out of functions if it works with only correct var derivatives
    # k is each relevant element
    def ReducedEQ0k(self, k, x):
        result = -x.b_i0[k] # -b_k0

        a_kj_n_J = x.gas_prod_a[k] * x.n_j[x.gas_prod_I[k]] # a_kj*n_j
        # print('ReducedEQ0k', k, x.subs[k].formula, a_kj_n_J.shape)
        for j, a_kj_n_j in zip(x.gas_prod_I[k], a_kj_n_J): # All substances containing this element
            result += np.sum(a_kj_n_j * x.subs_a[j] * x.pi_i[x.subs_I[j]]) # Sum across all elements in the substance, a_kj*a_ij*n_j*pi_i
            result += a_kj_n_j*x.Deltaln_n # a_kj*n_j*Deltaln_n
            H_D_j, S_D_j = x.subs[j].get_H_D(x.T), x.subs[j].get_S_D(x.T) # H_j/(R*T)
            result += a_kj_n_j*H_D_j*x.Deltaln_T # a_kj*n_j*H_j/(R*T)*Deltaln_T
            # TODO: check Gibb's formula!
            result -= a_kj_n_j*(H_D_j-S_D_j+np.log(x.n_j[j]/x.n)+np.log(x.P/1.0E5)) # a_kj*n_j*mu_j/(R*T)
            result += a_kj_n_j # b_k contribution, a_kj*n_j

        # sum across condensed, a_kj*Deltan_j
        # TODO: need to test if empty
        # result += np.sum(x.cond_prod_a[k] + x.Deltan_j[x.cond_prod_I[k] - x.N_gas])
        # result += x.cond_prod_a[k] * x.n_j[x.cond_prod_I[k]] # b_k contribution, a_kj*n_j
        # print('res shp', result.shape)

        return result

    # j is each condensed species (starting at j=gaseousSubstanceCount)
    def ReducedEQ1j(self, j, x):
        # TODO: How is Gibb's treated for condensed???
        # H_jstd/(R*T)*Deltaln_T-mu_j/(R*T)
        result = x.subs[j].get_H_D(x.T)*x.Deltaln_T-(x.subs[j].get_H_D(x.T)-x.subs[j].get_S_D(x.T))
        result += np.sum(subs_a[j] * x.pi_i[subs_I[j]]) # sum across elements, a_ij*pi_i

        return result

    def ReducedEQ2(self, x):
        result = -x.n - x.n*x.Deltaln_n
        for j in range(x.N_gas):
            result += np.sum(x.subs_a[j] * x.n_j[j] * x.pi_i[x.subs_I[j]]) # sum across i, a_ij*n_j*pi_i
            # TODO: SUS
            #result += (iter.n_j[j] - iter.n) * iter.Deltaln_n; # (n_j-n)*Deltaln_n
            result += x.n_j[j] * x.Deltaln_n; # (n_j-n)*Deltaln_n
            H_D_j, S_D_j = x.subs[j].get_H_D(x.T), x.subs[j].get_S_D(x.T) # H_j/(R*T)
            result += x.n_j[j]*H_D_j*x.Deltaln_T; # n_j*H_jstd/(R*T)*Deltaln_T
            result += x.n_j[j]; # n_j
            result -= x.n_j[j]*(H_D_j-S_D_j+np.log(x.n_j[j]/x.n)+np.log(x.P/1.0E5)); # n_j*mu_j/(R*T)

        return result

    def ReducedEQ3(self, x):
        result = -x.h_0/(Rhat*x.T) # h_0/R*T
        
        # TODO: could get rid of several loops by computing H, S, C_p ahead of time
        for j in range(x.N_gas):
            Cp_D, H_D_j, S_D_j = x.subs[j].get_Cp_D(x.T), x.subs[j].get_H_D(x.T), x.subs[j].get_S_D(x.T) # H_j/(R*T)
            # print(x.pi_i.shape, len(x.subs_I))
            result += np.sum(x.subs_a[j] * x.n_j[j] * H_D_j * x.pi_i[x.subs_I[j]]) # sum across i, a_ij*n_j*H_jstd/(R*T)*pi_i
            result += x.n_j[j] * H_D_j * x.Deltaln_n; # n_j*H_jstd/(R*T)*Deltaln_n
            result += x.n_j[j] * H_D_j; # h/(R*T) contribution, n_j*H_jstd/(R*T)
            result += x.n_j[j]*(Cp_D+H_D_j*H_D_j)*x.Deltaln_T; # (n_j*Cp_jstd/R+n_j*H_jstd^2/(R^2*T^2))*Deltaln_T
            result -= x.n_j[j]*H_D_j*(H_D_j-S_D_j+np.log(x.n_j[j]/x.n)+np.log(x.P/1.0E5)); # n_j*H_jstd*mu_j/(R^2*T^2)
        
        for j in range(x.N_gas, x.N_sub):
            H_D_j = x.subs[j].get_H_D(x.T) # H_j/(R*T)
            result += H_D_j*x.Deltan_j[j-x.N_gas]; # H_jstd/(R*T)*Deltan_j
            result += x.n_j[j] * H_D_j; # h/(R*T) contribution

        return result

    @dataclass
    class InternalState(object):
        present_elements: list[str]
        N_elem: int
        N_sub: int
        N_gas: int
        N_cond: int

        P: float # Pa, constant pressure
        h_0: float # input enthalpy
        n: float # kmol/kg, inverse molar mass of mixture
        T: float # K, current temperature
        Deltaln_n: float
        Deltaln_T: float

        # cond_subs: np.ndarray
        b_i0: np.ndarray # used to keep the elemental mass density balance identical to the inputs
        b_i0_max: float # (N_elem)
        pi_i: np.ndarray # (N_elem)
        n_j: np.ndarray # (N_subs), kmol of species j per kg of mixture
        Deltan_j: np.ndarray # (N_cond)
        Deltan_j_gas: np.ndarray # (N_gas)

        # TODO: sub should only check formula as we use .index here - not anymore
        subs: list[ThermoSubstance] # N_subs
        subs_I: list[np.ndarray] # N_subs array containing local element indices
        subs_a: list[np.ndarray] # Corresponding amount of the element

        gas_prod_I: list[np.ndarray] # N_elem arrays containing local substance indices
        gas_prod_a: list[np.ndarray] # corresponding amount of the element in the substance

        cond_prod_I: list[np.ndarray] # N_elem arrays containing local substance indices
        cond_prod_a: list[np.ndarray] # corresponding amount of the element in the substance
        
        def __init__(self, present_elements):
            self.present_elements = present_elements
    
    @dataclass
    class Result:
        T: float # K
        # Important: these are actual Cp, Cv, gamma; NOT FOZEN
        #   i.e. they account for changing chemical composiRtion w.r.t. pressure and density, respectively
        Cp: float
        Cv: float
        gamma: float # Specific heat ratio
        # Frozen specific heat ratio aka isentropic exponent
        #   used for speed of sound and anything else happening on a much smaller time scale than chemistry
        gamma_s: float
        M: float # kg/kmol
        R: float
        valid: bool # No errors and has any inputs
        iters: int
        composition: dict # Composition by (component formula, molar fraction)
        
        def __init__(self, valid):
            self.valid = valid

    # supply is a dict like {formula: (mass fraction, T in, P in)}
    def solve(self, Pc, supply, max_iters=200, internal_state=None, reinit=True):
        # Basic input checks
        if len(supply) == 0 or Pc <= 0.0:
            return Result(False), None
        
        # Electrons always included so that ions are possible to form
        present_elements = ['E']
        for formula, inputs in supply.items():
            substance = self.substances[formula]
            for elem, amount in substance.composition.items():
                if not elem in present_elements:
                    present_elements.append(elem)
        present_elements = sorted(present_elements)

        x = internal_state
        if x != None: # Check that provided internal_state is usable
            if present_elements != x.present_elements:
                print('Warning: provided internal state is not usable due to differing elemental composition, rebuilding...')
                x = None
        if x == None:
            x = self.InternalState(present_elements)
            x.N_elem = len(present_elements)
            x.gas_prod_I, x.gas_prod_a, x.cond_prod_I, x.cond_prod_a = [[[] for i in range(x.N_elem)] for i in range(4)]
            
            gas, cond = [], []
            for sub in self.substances.values():
                if sub.is_product:
                    # TODO: temp cutoff too like SSTS?
                    if not sub.condensed and all([elem in present_elements for elem in sub.composition]):
                        # print('RELEVANT GAS', sub.formula, sub.condensed)
                        gas.append(sub)
            x.subs = gas + cond
            x.N_sub, x.N_gas, x.N_cond = len(x.subs), len(gas), len(cond)

            x.subs_I, x.subs_a = [], []
            for i, sub in enumerate(x.subs):
                sub_I, sub_a = [], []
                for elem, amount in sub.composition.items():
                    j = present_elements.index(elem)
                    if i < x.N_gas:
                        
                        x.gas_prod_I[j].append(i)
                        x.gas_prod_a[j].append(amount)
                    else:
                        x.cond_prod_I[j].append(i)
                        x.cond_prod_a[j].append(amount)
                    sub_I.append(j)
                    sub_a.append(amount)

                x.subs_I.append(np.array(sub_I, dtype=int)), x.subs_a.append(np.array(sub_a, dtype=float))
            
            # Convert element to substance arrays to np arrays
            x.gas_prod_I, x.gas_prod_a, x.cond_prod_I, x.cond_prod_a = [[np.array(arr) for arr in coll] for coll in [x.gas_prod_I, x.gas_prod_a, x.cond_prod_I, x.cond_prod_a]]

            reinit = True

        if reinit:
            x.T = 3000 # K, current temperature
            x.n = 0.1 # total kmol/kg
            x.n_j = np.ones(x.N_gas) * 0.1 / x.N_gas
            if x.N_cond > 0.0:
                x.n_j = np.concatenate([x.n_j, np.zeros(x.N_cond)])
            x.pi_i = np.zeros(x.N_elem)
            x.Deltaln_n = 0.0
            x.Deltaln_T = 0.0
            x.Deltan_j = np.zeros(x.N_cond)
            x.Deltan_j_gas = np.zeros(x.N_gas)
            x.b_i0 = np.zeros(x.N_elem)
        
        # Begin from supply
        x.P = Pc
        x.h_0 = 0.0
        # print('all x subs', x.subs)
        for formula, inputs in supply.items():
            sub = self.substances[formula]
            m_frac, T, P = inputs
            # j = x.subs.index(sub)
            n_j = m_frac/sub.M
            x.h_0 += n_j * sub.get_H_D(T) * Rhat * T
            for elem, amount in sub.composition.items():
                x.b_i0[present_elements.index(elem)] += amount * n_j # Loop across elements
        x.b_i0_max = np.max(x.b_i0)

        N_dof = x.N_elem + x.N_cond + 2
        
        iter = 1
        while iter <= max_iters:
            rhs = np.zeros(N_dof)
            # equation 0, per element
            for k in range(x.N_elem):
                rhs[k] = self.ReducedEQ0k(k, x)
            # equation 1, per condensed product
            for j in range(x.N_gas, x.N_cond):
                rhs[j] = self.ReducedEQ1j(j, x)
            rhs[x.N_elem + x.N_cond]     = self.ReducedEQ2(x)
            rhs[x.N_elem + x.N_cond + 1] = self.ReducedEQ3(x)
            
            jac = np.zeros((N_dof, N_dof))
            for k in range(x.N_elem):
                # for (const std::pair<int, double>& ja : relatedGasousSubstances[k]) {
                for j, n in zip(x.gas_prod_I[k], x.gas_prod_a[k]):
                    H_D_j = x.subs[j].get_H_D(x.T)
                    # Per-element equation partial derivatives of pi_i, sum across k rows and i columns, a_kj*a_ij*n_j
                    jac[x.subs_I[j], k] += np.sum(n * x.subs_a[j] * x.n_j[x.subs_I[j]])
                    # Per-element equation partial derivative of Deltaln_T
                    jac[-1, k] += n * x.n_j[j] * H_D_j # a_kj*n_j*H_jstd/(R*T)
                    # Per-element equation partial derivative of Deltaln_n
                    jac[-2, k] += n * x.n_j[j] # a_kj*n_j
                # Per-element equation partial derivatives of Deltan_j
                if x.N_cond>0: jac[x.N_elem + (x.cond_prod_I[k] - x.N_gas), k] += x.cond_prod_a[k]
            for j in range(x.N_gas, x.N_sub):
                H_D_j = x.subs[j].get_H_D(x.T)
                # Per-condensed-species equation partial derivatives of pi_i
                if x.N_cond>0: jac[x.cond_prod_I[j], x.N_elem + (j - x.N_gas)] += x.cond_prod_a[j] # sum i, a_ij
                # Per-element equation partial derivative of Deltaln_T
                jac[-1, (x.N_elem + (j-x.N_gas))] = H_D_j
                # Enthalpy conservation equation partial derivatives of Deltan_j
                jac[x.N_elem + (j-x.N_gas), N-1] = H_D_j
            jac[-2, -2] = -x.n
            for j in range(x.N_gas):
                Cp_D, H_D_j = x.subs[j].get_Cp_D(x.T), x.subs[j].get_H_D(x.T)
                # Molar balance equation partial derivatives of pi_i
                jac[x.subs_I[j], -2] += x.subs_a[j] * x.n_j[j] # a_ij*n_j
                # Enthalpy conservation equation partial derivatives of pi_i
                jac[x.subs_I[j], -1] += x.subs_a[j] * x.n_j[j] * H_D_j # a_ij*n_j*H_jstd/(R*T)

                # Molar balance equation partial derivative of Deltaln_n
                jac[-2, -2] += x.n_j[j]
                # Molar balance equation partial derivative of Deltaln_T
                jac[-1, -2] += x.n_j[j]*H_D_j # n_j*H_jstd/(R*T)
            
                # Enthalpy conservation equation partial derivative of Deltaln_n
                jac[-2, -1] += x.n_j[j]*H_D_j # n_j*H_jstd/(R*T)
                # Enthalpy conservation equation partial derivative of Deltaln_T
                jac[-1, -1] += x.n_j[j]*(Cp_D+H_D_j**2) # (n_j*Cp_jstd/R+n_j*H_jstd^2/(R^2*T^2));

            # Newton method update
            # TODO: line search if singular?
            upd = np.linalg.solve(jac, rhs)
            x.pi_i -= upd[:x.N_elem]
            x.Deltan_j -= upd[x.N_elem:x.N_elem+x.N_cond]
            x.Deltaln_n -= upd[N_dof - 2]
            x.Deltaln_T -= upd[N_dof - 1]

            # Empirical lambda formulas suggested by NASA
            lambda1 = 5.0*np.max([np.abs(x.Deltaln_T), np.abs(x.Deltaln_n)])
            lambda2 = float('inf')
            for j in range(x.N_gas):
                H_D_j, S_D_j = x.subs[j].get_H_D(x.T), x.subs[j].get_S_D(x.T) # H_j/(R*T)
                x.Deltan_j_gas[j] = x.Deltaln_n + H_D_j*x.Deltaln_T - \
                    (H_D_j-S_D_j+np.log(x.n_j[j]/x.n)+np.log(x.P/1.0E5))
                x.Deltan_j_gas[j] += np.sum(x.subs_a[j] * x.pi_i[x.subs_I[j]]) # sum across i, a_ij*pi_i
                
                v = np.abs(x.Deltan_j_gas[j])
                if v > lambda1: lambda1 = v
                ln_nj_n = np.log(x.n_j[j]/x.n)
                if ln_nj_n <= -18.420681 and x.Deltan_j_gas[j] >= 0.0:
                    v = np.abs((-ln_nj_n-9.2103404) / (iter.Deltan_j_gas[j]-iter.Deltaln_n))
                    if v < lambda2:
                        lambda2 = v
            
            # Limit corrections to prevent diverging solutions due to large jumps
            lambda1 = 2.0 / lambda1
            lam = np.min([1.0, lambda1, lambda2])
            for j in range(x.N_gas):
                x.n_j[j] *= np.exp(lam * x.Deltan_j_gas[j])
            for j in range(x.N_gas, x.N_sub):
                x.n_j[j] = x.n_j[j] + lam*x.Deltan_j[j-x.N_gas]
            x.n *= np.exp(lam * x.Deltaln_n)
            x.T *= np.exp(lam * x.Deltaln_T)

            # Test for convergence, roughly testing the least expensive first
            if np.abs(x.Deltaln_T) <= 1.0E-4:
                sumN_j = np.sum(x.n_j)
                convergeGas = np.sum(x.n_j[:x.N_gas]*np.abs(x.Deltan_j_gas)) / sumN_j
                convergeCondensed = np.sum(np.abs(x.Deltan_j)) / sumN_j
                convergeTotal = x.n * np.abs(x.Deltaln_n) / sumN_j
                if convergeGas <= 0.5E-5 and convergeCondensed <= 0.5E-5 and convergeTotal <= 0.5E-5:
                    massBalanceConvergence = True
                    for i in range(x.N_elem):
                        sum_aij_nj = np.sum(x.gas_prod_a[i] * x.n_j[x.gas_prod_I[i]])
                        if np.abs(x.b_i0[i] - sum_aij_nj) > x.b_i0_max*1.0E-6:
                            massBalanceConvergence = False; break
                    if massBalanceConvergence: break
                    # TODO: Also do TRACE != 0 convergence test for pi_i

            iter += 1
        
        if iter == max_iters + 1: iter = max_iters # If terminated due to max_iters, will be 1 too high

        result = self.Result(True)
        result.T = x.T
        result.M = 1/x.n
        result.R = Rhat / result.M
        
        print(f'result T={result.T}, M={result.M}')

        N_dof = x.N_elem + x.N_cond + 1
        dP, dT = np.zeros(N_dof), np.zeros(N_dof)
        M = np.zeros((N_dof, N_dof)) # T and P deriv coeffs

        # dpi_i/dlnT then dn_j/dln_T then dlnn/dlnT
        # Per-element equation
        for k in range(x.N_elem):
            # TODO: Better way? > 1.0E-7 mainly prevents electron gas row from ruining the matrix
            for j, n in zip(x.gas_prod_I[k], x.gas_prod_a[k]):
                if x.n_j[j] > 1.0E-7:
                    M[x.subs_I[j], k] += n * x.n_j[j] * x.subs_I[j] # a_kj*a_ij*n_j
                    M[-1, k] += n * x.n_j[j] # a_kj*n_j
                    dT -= n * x.n_j[j] * x.subs[j].get_H_D(x.T)
                    dP += n * x.n_j[j]
            # TODO: rename n to a!
            for j, n in zip(x.cond_prod_I[k], x.cond_prod_a[k]):
                if x.n_j[j] > 1.0E-7:
                    M[x.N_elem + j, k] += n # a_kj
            # Basically make sure there are some substances with this element with a concentration larger than 1.0E-7
            if not np.any(M[:, k] != 0.0): # Just say that the derivative = 0.0
                M[k, k], dT[k], dP[k] = 1.0, 0.0, 0.0 # TODO: Why was this != 0.0, very small? - may have forgotten to = 0 in that old version
        # TODO: Something similar to above for very low concentrations?
        # Per-condensed equation
        for j in range(x.N_gas, x.N_sub):
            if x.n_j[j] > 1.0E-7:
                M[x.subs_I[j], N_elem+(j-x.N_gas)] += x.subs_a[j] # a_ij
                dT[x.N_elem+(j-x.N_gas)] += -x.subs[j].get_H_D(x.T) # -H_jstd/(R*T)
                # Note dP = 0 due to incompressible
        # Singular equation
        for j in range(x.N_gas):
            if x.n_j[j] > 1.0E-7:
                M[x.subs_I[j], -1] += x.n_j[j] * np.sum(x.subs_a[j]) # a_ij*n_j
                tempDerivs [-1] -= x.n_j[j] * x.subs[j].get_H_D(x.T) # n_j*H_jstd/(R*T)
                pressDerivs[-1] += x.n_j[j] # n_j

        dT, dP = np.linalg.solve(M, np.stack([dT, dP], axis=-1)).unstack(axis=-1)

        # TODO: If we are including condensed in Cp shouldn't they be in M too?
        result.Cp = np.sum([x.n_j[j] * x.subs[j].get_Cp_D(x.T) for j in range(x.N_sub)]) # Frozen contribution
        # TODO: Coeffs have already been computed in jacobian matrix, can we re-use?        
        for i in range(x.N_elem):
            for j, n in zip(x.gas_prod_I[i], x.gas_prod_a[i]):
                result.Cp += n * x.n_j[j] * x.subs[j].get_H_D(x.T)*dT[i];
        for j in range(x.N_gas, x.N_sub):
            result.Cp += x.subs[j].get_H_D(x.T)*dT[x.N_elem+(j-x.N_gas)];
        for j in range(x.N_gas):
            H_Dstd = x.subs[j].get_H_D(x.T);
            result.Cp += x.n_j[j]*(H_Dstd*dT[N_dof-1]+H_Dstd**2);
        result.Cp *= Rhat

        dlnV_dlnT = dT[N-1] + 1.0 # dlnn/dlnT+1
        dlnV_dlnP = dP[N-1] - 1.0 # dlnn/dlnP-1
        result.Cv      = result.Cp + result.R*dlnV_dlnT**2/dlnV_dlnP
        result.gamma   = result.Cp / result.Cv
        result.gamma_s = -result.gamma / dlnV_dlnP
        result.composition = { x.subs[j].formula: x.n_j[j]*result.M for j in range(x.N_sub) }

        return result, internal_state
