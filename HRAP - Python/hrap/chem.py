# HP Gibb's free energy minimization (specified input enthalpy and pressure)
# Note: RPA, CEA, PROPEP all use the same algorithm
# Why? Because pypropep is abandoned, pycea requires uses to install Fortran, and Cantera's equilibrium solver has been broken for over a decade. That is, there are no suitable Python alternatives.

# PORTING AND UNIT TESTS ARE WIP!



# from typing import Dict



# Terminlogy explanations:
# Mixture - reaction products

# Unit explanations:
# n_j - kmol of species j per kg of mixture
# s - specific entropy
# S - molar entropy
# h - specific enthalpy
# H - molar enthalpy
# g - Gibbs energy per kg of mixture
# mu_j - chemical potential per kmol of species j
# M - molecular weight
# DeltaHForm - enthalpy of formation

# Extension explanations:
# _0 - fixed/initial value
# ^0 - standard state value
# _D - dimensionless value

# Working equations:
# s = sum n_j*S_j
# h = sum n_j*H_j
# g = sum n_j*mu_j



Rhat = 8314 # J/(K*kmol), universal gas constant

@partial(jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[])
@dataclass
class NASA9(object) {
    lowerRange: float # K
    upperRange: float # K
    dho: float # TODO
    coeffs: list[float] # 9 coefficients
    
    # Nondimensionalized as Cp/R
    def get_Cp_D(self, T):
        return self.coeffs[0]/(T*T)      + self.coeffs[1]/T   + self.coeffs[2]      +
               self.coeffs[3]*T          + self.coeffs[4]*T*T + self.coeffs[5]*T**2 +
               self.coeffs[6]*T**4
    
    # Nondimensionalized as H/(R*T)
    def get_H_D(self, T):
         return -self.coeffs[0]/(T*T)          + self.coeffs[1]/T*jnp.log(T) + self.coeffs[2]      +
                 self.coeffs[3]*T/2.0          + self.coeffs[4]*T*T/3.0  + self.coeffs[5]*T**3/4.0 +
                 self.coeffs[6]*T**4/5.0 + self.coeffs[7]/T
    }

    # Nondimensionalized as S/R
    def get_S_D(self, T):
        return -self.coeffs[0]/(2.0*T*T)      - self.coeffs[1]/T       + self.coeffs[2]*jnp.log(T) +
                self.coeffs[3]*T              + self.coeffs[4]*T*T/2.0 + self.coeffs[5]*T**3/3.0   +
                self.coeffs[6]*T**4/4.0 + self.coeffs[8]
    }

@partial(jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[])
@dataclass
class ThermoSubstance(object):
    formula: str # Format should be similar to "(HCOOH)2-", all capitals
    comment: str # Such as data origin
    condensed: bool # Either condensed or gaseous
    isProduct: bool # Always a reactant, not always a product
    composition: Dict[str, float] # two-letter code: relative moles i.e. chemical composition
    M: float # kg/kmol, molar mass
    providers: list[NASA9]
    T_min: float # Range where the substance is allowed to be used (may be a single point, may contain error margin past where it is defined)
    T_max: float

    def get_R(self): # J/(K*kg)
        return Rhat / self.M
    
    def get_Cp_D(self, T):
    
    def get_H_D(self, T):
        
    def get_S_D(self, T):


@dataclass
class ChemSolver:
    substances: list[ThermoSubstance]
    
    # Takes in table such as thermo.ipa from RPA or thermo.dat from cpropep. Must be text, not binary
    # Must end with END REACTANTS and contain END PRODUCTS
    # Currently assumed to contain a section of comments, 3 lines without comments to be ignored, then data that may contain comments
    # Curve fit data should be dimensionless (Cp/R, S/R, and H/R) with S and H supplied at 1bar
    # Curve fit for Cp can not have more than 7 elements and exponents (-2, -1, 0, 1, 2, 3, 4) are always assumped, with data ignored
    def __init__(self, chem_file):
        
# Old C header
	
    # class ThermochemicalSubstance {
    # public:
        # class PROPEPSingleTempReactantProvider : public PropertyProvider {
        # public:
            # double T; // K
            # double DeltaHForm; // J/kmol

            # PROPEPSingleTempReactantProvider(double T, double DeltaHForm);

            # // Will return the same regardless of temperature, only valid if same as the single temperatuure
            # double GetH_D(double T) const;
        # };

        # static const PropertyProvider GLOBAL_DEFAULT_PROVIDER;
        # // For testing with singleTemperature
        # static const double DYNAMIC_TEMPERATURE;

        # std::string formula; // Format should be similar to "(HCOOH)2-", all capitals
        # std::string comment; // Such as data origin
        # // Either condensed or gaseous
        # bool condensed;
        # // Always a reactant, not always a product
        # bool isProduct;
        # std::vector<std::pair<Element::Symbol, double>> composition;
        # double M; // kg/kmol
        # const PropertyProvider* propertyProviders[PropertyProvider::COUNT];
        # // Either has a single temperature or a cp fit. If single temperature, cp usage is undefined
        # // double singleTemperature; // K
        # //PiecewisePolynomial fitProps[4]; // Pressure specific heat ratio, invalid if condensed, clamped within total range, nondimensionalized as Cp/R
        # // Range where the substance is allowed to be used (may be a single point, may contain error margin past where it is defined)
        # double T_min, T_max;

        # ThermochemicalSubstance();
        # ThermochemicalSubstance(ThermochemicalSubstance&& other);
        # virtual ~ThermochemicalSubstance();
        
        # // Specific R
        # inline double GetR() const { // J/(K*kg)
            # return Universal::R / M;
        # }

        # // Note that these use universal molar R, not specific
        # inline double GetCp_D(double T) const { // C_p/R
            # return propertyProviders[PropertyProvider::CP_D]->GetCp_D(T);
        # }

        # // For the standard state of 1bar
        # inline double GetH_D(double T) const { // H/(R*T)
            # return propertyProviders[PropertyProvider::H_D]->GetH_D(T);
        # }

        # // For the standard state of 1bar
        # inline double GetS_D(double T) const { // S/R
            # return propertyProviders[PropertyProvider::S_D]->GetS_D(T);
        # }

        # // TODO: Remove? I think this is for assigned entropy
        # // TODO: How can condensed species have these fits?!?!
        # //inline double GetS_D(double T, double P, double ln_nj_n) const // S/R
        # //{
        # //    if (condensed)
        # //        return GetS_D(T);
        # //    else
        # //        return GetS_D(T) - ln_nj_n - log(P/1.0E5);
        # //}

        # // TODO: ???
        # //inline double Getmu_D(double T, double P, double ln_nj_n) const // mu/R
        # //{
        # //    // Potential = total - unusable, love Jakob Ciciliano
        # //    if (condensed)
        # //        return GetH_D(T) - GetS_D(T);
        # //    else
        # //        return (GetH_D(T) - GetS_D(T)) + ln_nj_n + log(P/1.0E5);
        # //}

        # no_transfer_functions(ThermochemicalSubstance);
    # };

    # class ThermochemicalDatabase {
    # private:
        # // Needs to be a memory pool so that pointers are preserved
        # MemoryPool <ThermochemicalSubstance::PROPEPPiecewiseProvider> propepStyleFits;
        # std::vector<ThermochemicalSubstance> substances;
    # public:
		# ThermochemicalDatabase();
		
		# const std::vector<ThermochemicalSubstance> & GetAllSubstances() const;
	
        # // Such as thermo.ipa from RPA or thermo.dat from cpropep. Must be text, not binary
        # // Must end with END REACTANTS and contain END PRODUCTS
        # // Currently assumed to contain a section of comments, 3 lines without comments to be ignored, then data that may contain comments
        # // Curve fit data should be dimensionless (Cp/R, S/R, and H/R) with S and H supplied at 1bar
        # // Curve fit for Cp can not have more than 7 elements and exponents (-2, -1, 0, 1, 2, 3, 4) are always assumped, with data ignored
        # void LoadLegacyThermo(const std::string &directory);
        # ThermochemicalSubstance * GetSubstance(const std::string &formula);
    # };

    # /*
    # Gibbs free energy minimization, fixed enthalpy and pressure
    # Intended to be used with extended PROPEP data

    # Usage:
    # Create the class for the propulsion scheme and list all possible reactants
        # this will automatically create a list of all possible products and reactants in the scheme
            # only species whose constitutents are found in any combination of reactants are included
        # this cannot be changed without creating a new instance
    # Call Clear() once, then Supply() for each reactant being supplied in this time step
    # Call Solve() once
    # Repeat as needed Clear() through Solve()
    # */
    # class ThermochemicalSolverGibbsMinHP {
    # private:
        # struct ThermochemicalSubstanceInput {
            # ThermochemicalSubstance *substance;
            # double m_frac; // kg substance / kg total
            # double T; // K
            # double P; // Pa

            # ThermochemicalSubstanceInput(ThermochemicalSubstance *substance, double m_frac, double T, double P);
        # };

        # struct IterationVariables {
            # double P; // Pa, const
            # double h_0; // const
            # double n;
            # double Deltaln_n;
            # double T;
            # double Deltaln_T;
            # // This has the same size as substances
            # double *n_j; // kmol of substance / kg of mixture
            # // This has the size of subtsances minus gaseousSubstanceCount
            # double *Deltan_j;
            # // These have the same size as elements
            # double *pi_i;
            # double *b_i0; // const
            # double b_i0_max;

            # // Computed after the above
            # double *Deltan_j_gas; // Size of gaseousSubstanceCount
            # double *jacobian;
            # double *funcEvals;
            # int *ipiv;
        # };
        
        # std::vector<ThermochemicalSubstance *> substances;
        # int gaseousSubstanceCount; // Any subsequent entries are condensed
        # std::vector<Element::Symbol> elements; // local to global (usage ID to periodic table)
        # int reverseElements[Element::COUNT]; // global to local
        # // Local element to array of local substances (and relevant composition coefficient), used for the per-element equation
		# // TODO: Why did I make this a pointer?
        # std::vector<std::pair<int, double>> *relatedGasousSubstances;
        # std::vector<std::pair<int, double>> *relatedCondensedSubstances;
        # // May contain substances not in the local subtances list if they are reactant-only
        # std::vector<ThermochemicalSubstanceInput> supply;
        # IterationVariables iter;

        # double ReducedEQ0k(int k);
        # double ReducedEQ1j(int j);
        # double ReducedEQ2();
        # double ReducedEQ3(); // Enthalpy conservation
    # public:
        # struct Result {
        # public:
            # double T; // K
            # double Cp;
            # double Cv;
            # double gamma; // Specific heat ratio
            # // Used for speed of sound and ???
            # double gamma_s; // Isentropic exponent
            # double M; // kg/kmol
            # double R;
            # bool valid; // No errors and has any inputs
            # int iterations;
            # // Composition by molar fraction
            # std::vector<std::pair<ThermochemicalSubstance*, double>> composition;
        # };

        # ThermochemicalSolverGibbsMinHP();
        # ThermochemicalSolverGibbsMinHP(const std::vector<ThermochemicalSubstance *> &possibleReactants, const ThermochemicalDatabase &database);
        # ThermochemicalSolverGibbsMinHP & operator=(ThermochemicalSolverGibbsMinHP &&other);
        # ~ThermochemicalSolverGibbsMinHP();

        # // If persistent, must still be resupplied, but last solution will be the inital guess!
        # void Clear(bool persistentSolution);
        # //void Supply(ThermochemicalSubstance* substance, double m_frac, double T); // mass fraction, T(K)
        # void Supply(ThermochemicalSubstance *substance, double m_frac, double T, double P); // mass fraction, T(K), P(Pa)
        # Result Solve(double P, int iterations, bool returnComposition = false); // P(Pa)

        # no_transfer_functions(ThermochemicalSolverGibbsMinHP);
    # };



# OLD C Body
# namespace SSTS {
    # ThermochemicalSubstance::PROPEPPiecewiseProvider::PROPEPPiecewiseProvider() :
        # PropertyProvider(false),
        # lowerRange(HUGE_VAL),
        # upperRange(HUGE_VAL),
        # coefficients{0.0}
    # { }

    # ThermochemicalSubstance::PROPEPSingleTempReactantProvider::PROPEPSingleTempReactantProvider(double T, double DeltaHForm) :
        # PropertyProvider(true),
        # T(T),
        # DeltaHForm(DeltaHForm)
    # { }

    # double ThermochemicalSubstance::PROPEPSingleTempReactantProvider::GetH_D(double T) const {
        # return DeltaHForm / (Universal::R * T);
    # }

    # const ThermochemicalSubstance::PropertyProvider ThermochemicalSubstance::GLOBAL_DEFAULT_PROVIDER(false);
    # const double ThermochemicalSubstance::DYNAMIC_TEMPERATURE = -1.0;

    # ThermochemicalSubstance::ThermochemicalSubstance() :
        # formula("null"),
        # comment("undefined"),
        # condensed(false),
        # isProduct(false),
        # M(0.0),
        # T_min(-HUGE_VAL),
        # T_max( HUGE_VAL)
	# {
        # for (int i = 0; i < PropertyProvider::COUNT; ++i)
            # propertyProviders[i] = &GLOBAL_DEFAULT_PROVIDER;
    # }

    # ThermochemicalSubstance::ThermochemicalSubstance(ThermochemicalSubstance &&other) :
        # formula(std::move(other.formula)),
        # comment(std::move(other.comment)),
        # condensed(other.condensed),
        # isProduct(other.isProduct),
        # composition(std::move(other.composition)),
        # M(other.M),
        # T_min(other.T_min),
        # T_max(other.T_max)
	# {
        # for (int i = 0; i < PropertyProvider::COUNT; ++i) {
            # propertyProviders[i] = other.propertyProviders[i];
            # other.propertyProviders[i] = &GLOBAL_DEFAULT_PROVIDER;
        # }
    # }

    # ThermochemicalSubstance::~ThermochemicalSubstance() {
        # for (int i = 0; i < PropertyProvider::COUNT; ++i)
            # if (propertyProviders[i]->automanaged)
                # delete propertyProviders[i];
    # }

    # ThermochemicalDatabase::ThermochemicalDatabase() { }

    # namespace {
        # bool StartsWith(const std::string &str, const std::string &smallString) {
            # if (str.size() < smallString.size())
                # return false;

            # for (size_t i = 0; i < smallString.size(); ++i)
                # if (str[i] != smallString[i])
                    # return false;

            # return true;
        # }

        # bool ReadLine(std::ifstream &file, std::string &line) {
            # if (!file.good())
                # return false;

            # getline(file, line);
            # if (line[0] == '!')
                # return ReadLine(file, line);

            # return true;
        # }

        # std::string Trim(const std::string &str) {
            # if (str[0] != ' ' && str[str.length()-1] != ' ')
                # return str;

            # int first;
            # int last;
            # for (first = 0; first < str.length(); ++first)
                # if (str[first] != ' ')
                    # break;
            # if (first == str.length())
                # return "";
            # for (last = str.length()-1; last > 0; --last)
                # if (str[last] != ' ')
                    # break;

            # return str.substr(first, last-first+1);
        # }
    # }

    # void ThermochemicalDatabase::LoadLegacyThermo(const std::string &directory) {
        # /* Reads from entries such as:
        # name (18 char), comment (all remaining on line)
        # H2O               Hf:Cox,1989. Woolley,1987. TRC(10/88) tuv25.                  
        # temperature range entry count (2 char, kelvin), source identifier (8x char), elements (5x [2x char element, 6x char quantity]]), phase (2char, 0 for gas 1 for condensed), molecular weight (13 char, kg/kmol), heat of formation (15 char, J/mol, or enthalpy if entry count = 0 [single temperature, condensed])
         # 2 g 8/89 H   2.00O   1.00    0.00    0.00    0.00 0   18.0152800    -241826.000
        # temperature interval (2x 11 char), ncoeffs (1 char), exponents (8x 5 char), H^O(298.15)-H^O(0) (17 char, ?)
            # 200.000   1000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         9904.092
        # coefficients (across 2 lines, up to 8x 16 char, may be blank to imply skip for formatting)
        # -3.947960830D+04 5.755731020D+02 9.317826530D-01 7.222712860D-03-7.342557370D-06
         # 4.955043490D-09-1.336933246D-12                -3.303974310D+04 1.724205775D+01
        # (repeated for other temperature intervals)
           # 1000.000   6000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         9904.092
         # 1.034972096D+06-2.412698562D+03 4.646110780D+00 2.291998307D-03-6.836830480D-07
         # 9.426468930D-11-4.822380530D-15                -1.384286509D+04-7.978148510D+00
        # */

        # std::ifstream file;
	    # file.open(directory.c_str());
        # std::string line;
		
		# if (!file.is_open()) {
			# Log(Severity::ERROR, std::format("Unable to open thermo file: {}", directory));
			# return;
		# }

		# while (file.good()) {
			# getline(file, line);
			# if (line[0] != '!')
				# break;
		# }

		# for (int i = 0; i < 3; ++i)
			# if (!ReadLine(file, line))
				# return;

		# bool areProducts = true;
		# ThermochemicalSubstance substance;
		# while (file.good()) {
			# //std::cerr << line << std::endl;
			# if (StartsWith(line, "END PRODUCTS")) {
				# areProducts = false;
				# if (!ReadLine(file, line))
					# break;
				# continue;
			# }
			# if (StartsWith(line, "END REACTANTS"))
				# break;
			
			# substance.isProduct = areProducts;
			# int i = 0;
			# substance.formula = Trim(line.substr(i, 18)); i += 18;
			# if (line.length() > 18)
				# substance.comment = Trim(line.substr(i));
			# if (!ReadLine(file, line))
				# break;
			# i = 0;
			# int fitPieces = std::stoi(line.substr(i, 2)); i += 2;
			# i += 8; // Ignore source identifier
			# for (int j = 0; j < 5; ++j) {
				# std::string symbol = Trim(line.substr(i, 2)); i += 2;
				# double quantity = std::stod(line.substr(i, 6)); i += 6;
				# if (symbol.length() == 0)
					# continue;
				# if (symbol.length() > 1 && symbol[1] >= 'A' && symbol[1] <= 'Z')
					# symbol[1] = symbol[1] + 32; // Uppercase to lowercase
				# if (PeriodicTable::HasSymbolShort(symbol))
					# substance.composition.push_back(std::pair<Element::Symbol, double>(PeriodicTable::GetSymbolShort(symbol), quantity));
				# else {
					# Log(Severity::ERROR, std::format("Element {} is not recognized in {}", symbol, directory));
					# substance.composition.push_back(std::pair<Element::Symbol, double>(Element::Symbol::COUNT, quantity));
				# }
			# }
			# int phase = std::stoi(line.substr(i, 2)); i += 2;
			# if (phase == 0)
				# substance.condensed = false;
			# else
				# substance.condensed = true;
			# substance.M = std::stod(line.substr(i, 13)); i += 13;
			# // Is this always DeltaHForm (if so for what temp)? Or is it sometimes only H(298)-H(0) for some reason (or is that comperable, as to have no change temps must be different - no because always same across 2 at 0 if so which it isn't)? - Seems to be enthalpy of formation always, see H2O(L)
			# // Comes as kJ/kmol so get to J/kmol
			# double DeltaHForm = 1000.0 * std::stod(line.substr(i, 15));
			# if (!ReadLine(file, line))
				# break;
			# // If single temp, usually because it is condensed
			# if (fitPieces == 0) {
				# if (areProducts) {
					# Log(Severity::ERROR, std::format("Zero piece data is not permitted for products, {} in {}, as entropy is needed to solve!", substance.formula, directory));
					# return;
				# }
				# double singleTemperature = std::stod(line.substr(i, 11));
				# substance.propertyProviders[ThermochemicalSubstance::PropertyProvider::H_D] = new ThermochemicalSubstance::PROPEPSingleTempReactantProvider(singleTemperature, DeltaHForm);
				# substance.T_min = singleTemperature * 0.9;
				# substance.T_max = singleTemperature * 1.1;

				# if (!ReadLine(file, line))
					# break;
			# } else {
				# //substance.singleTemperature = ThermochemicalSubstance::DYNAMIC_TEMPERATURE;
				# substance.T_min = HUGE_VAL;
				# substance.T_max = -HUGE_VAL;
				# ThermochemicalSubstance::PROPEPPiecewiseProvider *lastDatafit = nullptr;
				# for (int j = 0; j < fitPieces; ++j) {
					# ThermochemicalSubstance::PROPEPPiecewiseProvider *datafit = propepStyleFits.alloc();
					# if (j == 0) {
						# substance.propertyProviders[ThermochemicalSubstance::PropertyProvider::CP_D] = datafit;
						# substance.propertyProviders[ThermochemicalSubstance::PropertyProvider::H_D]  = datafit;
						# substance.propertyProviders[ThermochemicalSubstance::PropertyProvider::S_D]  = datafit;
					# } else {
						# lastDatafit->next = datafit;
					# }
					# lastDatafit = datafit;

					# i = 0;
					# datafit->lowerRange = std::stod(line.substr(i, 11)); i += 11;
					# datafit->upperRange = std::stod(line.substr(i, 11)); i += 11;
					# if (datafit->lowerRange < substance.T_min)
						# substance.T_min = datafit->lowerRange;
					# if (datafit->upperRange > substance.T_max)
						# substance.T_max = datafit->upperRange;
					# int ncoffs = std::stoi(line.substr(i, 1)); i += 1;
					# if (ncoffs > 8) {
						# Log(Severity::ERROR, std::format("Thermochemical substance {} in {} cannot have more than 8 coefficients!", substance.formula, directory));
						# return;
					# }
					# // Ignore exponents (only specified for C_p and have to make consistent assumptions for others)
					# i += 8*5;
					# //for (int k = 0; k < 8; ++k)
					# //    substance.fitProps[j].exponents[k] = std::stod(line.substr(i, 5)); i += 5;
					# // TODO: Why is this tabulated? Should already be obtainable using fits... 
					# datafit->dho = std::stod(line.substr(i, 17));
					# // Should always have 2 lines. Second line should have maximum 2 data slots on the left and exactly 2 on the right
					# if (!ReadLine(file, line))
						# return;
					# int m = 0;
					# i = 0;
					# for (int k = 0; k < 5; ++k) {
						# std::string coeff = Trim(line.substr(i, 16)); i += 16;
						# if (coeff == "")
							# continue;
						# for (int s = coeff.size()-1; s >= 0; --s)
							# if (coeff[s] == 'D') {
								# coeff[s] = 'E';
								# break;
							# }

						# datafit->coefficients[m] = std::stod(coeff);
						# ++m;
					# }
					# if (!ReadLine(file, line))
						# return;
					# i = 0;
					# for (int k = 0; k < 2; ++k) {
						# std::string coeff = Trim(line.substr(i, 16)); i += 16;
						# if (coeff == "")
							# continue;
						# for (int s = coeff.size()-1; s >= 0; --s)
							# if (coeff[s] == 'D') {
								# coeff[s] = 'E';
								# break;
							# }
						# datafit->coefficients[m] = std::stod(coeff);
						# ++m;
					# }
					# // Center is always ignored
					# i += 16;
					# // Last two are for dimensionless molar enthalpy and entropy, respectively
					# for (int k = 0; k < 2; ++k) {
						# std::string coeff = Trim(line.substr(i, 16)); i += 16;
						# if (coeff == "")
							# continue;
						# for (int s = coeff.size()-1; s >= 0; --s)
							# if (coeff[s] == 'D') {
								# coeff[s] = 'E';
								# break;
							# }
						# datafit->coefficients[7+k] = std::stod(coeff);
					# }
					# if (!ReadLine(file, line))
						# return;
				# }
			# }

			# // TODO: Error if already exists
			# substances.push_back(std::move(substance));
		# }
    # }

    # ThermochemicalSubstance* ThermochemicalDatabase::GetSubstance(const std::string& formula) {
        # for (ThermochemicalSubstance& chem : substances)
            # if (chem.formula == formula)
                # return &chem;
        # return nullptr;
    # }

    # const std::vector<ThermochemicalSubstance> & ThermochemicalDatabase::GetAllSubstances() const {
        # return substances;
    # }

    # ThermochemicalSolverGibbsMinHP::ThermochemicalSubstanceInput::ThermochemicalSubstanceInput(ThermochemicalSubstance* substance, double m_frac, double T, double P) :
        # substance(substance),
        # m_frac(m_frac),
        # T(T),
        # P(P)
    # { }

    # ThermochemicalSolverGibbsMinHP::ThermochemicalSolverGibbsMinHP() :
        # relatedGasousSubstances   (nullptr),
        # relatedCondensedSubstances(nullptr)
    # { }

    # ThermochemicalSolverGibbsMinHP::ThermochemicalSolverGibbsMinHP(const std::vector<ThermochemicalSubstance *> &possibleReactants, const ThermochemicalDatabase &database) {
        # for (int i = 0; i < Element::COUNT; ++i)
            # reverseElements[i] = Element::COUNT;
        # bool presentElements[Element::COUNT] = { false };
        # // Always add electrons to always include ions
        # presentElements[Element::E] = true;
        # reverseElements[Element::E] = 0;
        # elements.push_back(Element::E);
        # for (ThermochemicalSubstance *chem : possibleReactants)
            # for (const std::pair<Element::Symbol, double> &elem : chem->composition)
                # if (!presentElements[elem.first]) {
                    # presentElements[elem.first] = true;
                    # reverseElements[elem.first] = elements.size();
                    # elements.push_back(elem.first);
                # }
        # std::vector<ThermochemicalSubstance*> potentialCondensedState;
        # for (const ThermochemicalSubstance &chem : database.GetAllSubstances())
            # if (chem.isProduct) {
                # bool relevant = true;
                # for (const std::pair<Element::Symbol, double> &elem : chem.composition)
                    # relevant &= presentElements[elem.first];
                # if (relevant && (!(chem.T_min <= 2000.0 && chem.T_max >= 4000.0) || chem.condensed)) {
                    # //std::cerr << "Substance " << chem.formula << " has been elimated for debugging due to inadequate temperature range and/or for being condensed" << std::endl;
                    # relevant = false;
                # }
                # if (relevant) {
                    # //std::cerr << chem.formula << " was deemed relevant" << std::endl;
                    # if (chem.condensed)
                        # potentialCondensedState.push_back(const_cast<ThermochemicalSubstance*>(&chem));
                    # else
                        # substances.push_back(const_cast<ThermochemicalSubstance*>(&chem));
                # }
            # }
        # gaseousSubstanceCount = substances.size();
        # for (ThermochemicalSubstance* chem : potentialCondensedState)
            # substances.push_back(chem);

        # relatedGasousSubstances = new std::vector<std::pair<int, double>>[elements.size()];
        # for (int j = 0; j < gaseousSubstanceCount; ++j)
            # for (const std::pair<Element::Symbol, double>& elem : substances[j]->composition)
                # relatedGasousSubstances[reverseElements[elem.first]].push_back(std::pair<int, double>(j, elem.second));
        # relatedCondensedSubstances = new std::vector<std::pair<int, double>>[elements.size()];
        # for (int j = gaseousSubstanceCount; j < substances.size(); ++j)
            # for (const std::pair<Element::Symbol, double>& elem : substances[j]->composition)
                # relatedCondensedSubstances[reverseElements[elem.first]].push_back(std::pair<int, double>(j, elem.second));
        
        # iter.n_j = new double[substances.size()];
        # if (substances.size() > gaseousSubstanceCount)
            # iter.Deltan_j = new double[substances.size() - gaseousSubstanceCount];
        # else
            # iter.Deltan_j = nullptr;
        # iter.pi_i = new double[elements.size()];
        # iter.b_i0 = new double[elements.size()];

        # iter.Deltan_j_gas = new double[gaseousSubstanceCount];
        # int N = elements.size() + (substances.size()-gaseousSubstanceCount) + 2;
        # iter.jacobian  = new double[N*N];
        # // Make large enough to store elements.size() + condensedSubstanceCount + 1 for derivative RHS
        # iter.funcEvals = new double[2*(N-1)];
        # iter.ipiv      = new int[N];

        # iter.T = -1.0; // Mark as uninitialized, Clear() will handle
    # }

    # ThermochemicalSolverGibbsMinHP & ThermochemicalSolverGibbsMinHP::operator=(ThermochemicalSolverGibbsMinHP &&other) {
        # substances = std::move(other.substances);
        # gaseousSubstanceCount = other.gaseousSubstanceCount;
        # elements = std::move(other.elements);
        # for (int i = 0; i < Element::COUNT; ++i)
            # reverseElements[i] = other.reverseElements[i];
        # relatedGasousSubstances = other.relatedGasousSubstances;
        # relatedCondensedSubstances = other.relatedCondensedSubstances;
        # supply = std::move(other.supply);
        # iter = other.iter;

        # other.relatedGasousSubstances = nullptr; // Used in the destructor

        # return *this;
    # }

    # ThermochemicalSolverGibbsMinHP::~ThermochemicalSolverGibbsMinHP() {
        # if (relatedGasousSubstances != nullptr) { // Signals all others
            # delete[] relatedGasousSubstances;
            # delete[] relatedCondensedSubstances;
            # delete[] iter.n_j;
            # if (substances.size() > gaseousSubstanceCount)
                # delete[] iter.Deltan_j;
            # delete[] iter.pi_i;
            # delete[] iter.b_i0;
            # delete[] iter.Deltan_j_gas;
            # delete[] iter.jacobian;
            # delete[] iter.funcEvals;
            # delete[] iter.ipiv;
        # }
    # }

    # void ThermochemicalSolverGibbsMinHP::Clear(bool persistentSolution) {
        # supply.clear();
        # if (!persistentSolution || iter.T == -1.0) { // First time or voluntary reset
            # // Similar to original CEA first guesses
            # iter.T = 3000; // K
            # iter.n = 0.1; // total kmol/kg
            # // TODO: n_j for condensed is undefined! 
            # for (int j = 0; j < gaseousSubstanceCount; ++j)
                # iter.n_j[j] = 0.1 / gaseousSubstanceCount;
            # for (int j = gaseousSubstanceCount; j < substances.size(); ++j)
                # iter.n_j[j] = 0.0;
            # for (int i = 0; i < elements.size(); ++i)
                # iter.pi_i[i] = 0.0;
            # iter.Deltaln_n = 0.0;
            # iter.Deltaln_T = 0.0;
            # for (int j = 0; j < substances.size() - gaseousSubstanceCount; ++j)
                # iter.Deltan_j[j] = 0.0;
        # }
    # }

    # /*void ThermochemicalSolverGibbsMinHP::Supply(ThermochemicalSubstance* substance, double m_frac, double T)
    # {
        # // TODO: Put in saturated P (if available)!
        # supply.push_back(ThermochemicalSubstanceInput(substance, m_frac, T, 1.0E5));
    # }*/

    # void ThermochemicalSolverGibbsMinHP::Supply(ThermochemicalSubstance *substance, double m_frac, double T, double P) {
        # supply.push_back(ThermochemicalSubstanceInput(substance, m_frac, T, P));
    # }

    # // TODO: Take out of functions if it works with only correct var derivatives
    # // k is each relevant element
    # double ThermochemicalSolverGibbsMinHP::ReducedEQ0k(int k) {
        # double result = -iter.b_i0[k]; // -b_k0
        # for (const std::pair<int, double> &ja : relatedGasousSubstances[k]) {
            # double a_kj_n_j = ja.second * iter.n_j[ja.first]; // a_kj*n_j
            # for (auto &elem_i : substances[ja.first]->composition) // sum across i, a_kj*a_ij*n_j*pi_i
                # result += a_kj_n_j*elem_i.second*iter.pi_i[reverseElements[elem_i.first]];
            # result += a_kj_n_j*iter.Deltaln_n; // a_kj*n_j*Deltaln_n
            # double H_D_j = substances[ja.first]->GetH_D(iter.T); // H_jstd/(R*T)
            # result += a_kj_n_j*H_D_j*iter.Deltaln_T; // a_kj*n_j*H_jstd/(R*T)*Deltaln_T
            # // TODO: Double check Gibb's formula!
            # result -= a_kj_n_j*(H_D_j-substances[ja.first]->GetS_D(iter.T)+log(iter.n_j[ja.first]/iter.n)+log(iter.P/1.0E5)); // a_kj*n_j*mu_j/(R*T)
            # result += a_kj_n_j; // b_k contribution, a_kj*n_j
        # }
        # for (const std::pair<int, double> &ja : relatedCondensedSubstances[k]) { // sum across condensed, a_kj*Deltan_j
            # result += ja.second + iter.Deltan_j[ja.first-gaseousSubstanceCount];
            # // b_k contribution
            # result += ja.second * iter.n_j[ja.first]; // b_k contribution, a_kj*n_j
        # }

        # return result;
    # }

    # // j is each condensed species (starting at j=gaseousSubstanceCount)
    # double ThermochemicalSolverGibbsMinHP::ReducedEQ1j(int j) {
        # // TODO: How is Gibb's treated for condensed???
        # // H_jstd/(R*T)*Deltaln_T-mu_j/(R*T)
        # double result = substances[j]->GetH_D(iter.T)*iter.Deltaln_T-(substances[j]->GetH_D(iter.T)-substances[j]->GetS_D(iter.T));
        # for (auto &elem_i : substances[j]->composition) // sum across i, a_ij*pi_i
            # result += elem_i.second*iter.pi_i[reverseElements[elem_i.first]];

        # return result;
    # }

    # double ThermochemicalSolverGibbsMinHP::ReducedEQ2() {
        # double result = -iter.n - iter.n*iter.Deltaln_n;
        # for (int j = 0; j < gaseousSubstanceCount; ++j) {
            # for (auto &elem_i : substances[j]->composition) { // sum across i, a_ij*n_j*pi_i
                # result += elem_i.second*iter.n_j[j]*iter.pi_i[reverseElements[elem_i.first]];
                # //std::cerr << "elem " << elem_i.first << " " << iter.pi_i[reverseElements[elem_i.first]] << std::endl;
            # }
            # // TODO: SUS
            # //result += (iter.n_j[j] - iter.n) * iter.Deltaln_n; // (n_j-n)*Deltaln_n
            # result += iter.n_j[j] * iter.Deltaln_n; // (n_j-n)*Deltaln_n
            # double H_D_j = substances[j]->GetH_D(iter.T); // H_jstd/(R*T)
            # result += iter.n_j[j]*H_D_j*iter.Deltaln_T; // n_j*H_jstd/(R*T)*Deltaln_T
            # result += iter.n_j[j]; // n_j
            # result -= iter.n_j[j]*(H_D_j-substances[j]->GetS_D(iter.T)+log(iter.n_j[j]/iter.n)+log(iter.P/1.0E5)); // n_j*mu_j/(R*T)
            # //std::cerr << "substance " << j << " " << iter.n_j[j] << " " << H_D_j << " " << substances[j]->GetS_D(iter.T) << " " << iter.n << " " << iter.Deltaln_n << " " << iter.Deltaln_T << std::endl;
        # }

        # return result;
    # }

    # double ThermochemicalSolverGibbsMinHP::ReducedEQ3() {
        # double result = -iter.h_0/(Universal::R*iter.T); // h_0/R*T
        
        # for (int j = 0; j < gaseousSubstanceCount; ++j) {
            # double H_D_j = substances[j]->GetH_D(iter.T); // H_jstd/(R*T)
            # for (auto &elem_i : substances[j]->composition) // sum across i, a_ij*n_j*H_jstd/(R*T)*pi_i
                # result += elem_i.second*iter.n_j[j]*H_D_j*iter.pi_i[reverseElements[elem_i.first]];
            # result += iter.n_j[j] * H_D_j * iter.Deltaln_n; // n_j*H_jstd/(R*T)*Deltaln_n
            # result += iter.n_j[j] * H_D_j; // h/(R*T) contribution, n_j*H_jstd/(R*T)
            # result += iter.n_j[j]*(substances[j]->GetCp_D(iter.T)+H_D_j*H_D_j)*iter.Deltaln_T; // (n_j*Cp_jstd/R+n_j*H_jstd^2/(R^2*T^2))*Deltaln_T
            # result -= iter.n_j[j]*H_D_j*(H_D_j-substances[j]->GetS_D(iter.T)+log(iter.n_j[j]/iter.n)+log(iter.P/1.0E5)); // n_j*H_jstd*mu_j/(R^2*T^2)
            # //std::cerr << "lasteq,for,in: " << j << ", " << H_D_j << " " << iter.n_j[j] << std::endl;
        # }
        # //std::cerr << "lasteq,gas: " << result << std::endl;
        # for (int j = gaseousSubstanceCount; j < substances.size(); ++j) {
            # double H_D_j = substances[j]->GetH_D(iter.T);
            # result += H_D_j*iter.Deltan_j[j-gaseousSubstanceCount]; // H_jstd/(R*T)*Deltan_j
            # result += iter.n_j[j] * H_D_j; // h/(R*T) contribution
        # }

        # return result;
    # }

    # ThermochemicalSolverGibbsMinHP::Result ThermochemicalSolverGibbsMinHP::Solve(double P, int iterations, bool returnComposition) {
        # if (supply.empty()) {
            # Result result;
            # result.valid = false;

            # return result;
        # }
        # Result result;
        # result.valid = true;
        # result.iterations = 0;
        # // TODO: Deltaln_n, Deltaln_T, Deltan_j initialization!
        # iter.P = P;

        # // Specific initial (fixed) enthalpy
        # iter.h_0 = 0.0;
        # for (int i = 0; i < elements.size(); ++i)
            # iter.b_i0[i] = 0.0;
        # // Compute input molar fractions from mass inputs
        # for (ThermochemicalSubstanceInput &input : supply) {
            # double n_j = input.m_frac/input.substance->M; //(input.m/input.substance->M)/m;
            # //std::cerr << "input: " << input.substance->formula << " " << n_j << " " << input.T << " " << input.P << std::endl;
            # iter.h_0 += n_j * input.substance->GetH_D(input.T) * Universal::R * input.T;
            # for (const std::pair<Element::Symbol, double> &elem : input.substance->composition)
                # // TODO: Should just be 2 kgatoms/kmol of H2O right?
                # // b_i0 is used to keep the elemental mass density balance identical to the inputs
                # iter.b_i0[reverseElements[elem.first]] += elem.second * n_j;
        # }
        # iter.b_i0_max = 0.0;
        # for (int i = 0; i < elements.size(); ++i) if (iter.b_i0[i] > iter.b_i0_max) iter.b_i0_max = iter.b_i0[i];

        # int condensedSubstanceCount = substances.size() - gaseousSubstanceCount;
        # int N = elements.size() + condensedSubstanceCount + 2;
        # int nrhs = 1; // Only one rhs
        # int errorCode;
        
        # for (int iteration = 0; iteration < iterations; ++iteration) {
            # // equation 0, per element
            # for (int k = 0; k < elements.size(); ++k)
                # iter.funcEvals[k] = ReducedEQ0k(k);
            # // equation 1, per condensed product
            # for (int j = 0; j < condensedSubstanceCount; ++j)
                # iter.funcEvals[j + elements.size()] = ReducedEQ1j(j + gaseousSubstanceCount);
            # iter.funcEvals[elements.size() + condensedSubstanceCount]     = ReducedEQ2();
            # iter.funcEvals[elements.size() + condensedSubstanceCount + 1] = ReducedEQ3();
            
            # // Initialize with 0, only fill non-zero elements
            # for (int n = 0; n < N*N; ++n) iter.jacobian[n] = 0.0;
            # for (int k = 0; k < elements.size(); ++k) {
                # for (const std::pair<int, double>& ja : relatedGasousSubstances[k]) {
                    # // Per-element equation partial derivatives of pi_i, sum across k rows and i columns, a_kj*a_ij*n_j
                    # for (auto &elem_i : substances[ja.first]->composition)
                        # iter.jacobian[k + reverseElements[elem_i.first]*N] += ja.second*elem_i.second*iter.n_j[ja.first];
                    # // Per-element equation partial derivative of Deltaln_n
                    # iter.jacobian[k + (N-2)*N] += ja.second*iter.n_j[ja.first]; // a_kj*n_j
                    # // Per-element equation partial derivative of Deltaln_T
                    # iter.jacobian[k + (N-1)*N] += ja.second*iter.n_j[ja.first]*substances[ja.first]->GetH_D(iter.T); // a_kj*n_j*H_jstd/(R*T)
                # }
                # // Per-element equation partial derivatives of Deltan_j
                # for (const std::pair<int, double>& ja : relatedCondensedSubstances[k])
                    # iter.jacobian[k + (elements.size() + (ja.first-gaseousSubstanceCount))*N] += ja.second;
            # }
            # for (int j = gaseousSubstanceCount; j < substances.size(); ++j) {
                # // Per-condensed-species equation partial derivatives of pi_i
                # for (auto &elem_i : substances[j]->composition) // sum i, a_ij
                    # iter.jacobian[(elements.size() + (j-gaseousSubstanceCount)) + reverseElements[elem_i.first]*N] += elem_i.second;
                # double H_D_j = substances[j]->GetH_D(iter.T);
                # // Per-element equation partial derivative of Deltaln_T
                # iter.jacobian[(elements.size() + (j-gaseousSubstanceCount)) + (N-1)*N] = H_D_j;

                # // Enthalpy conservation equation partial derivatives of Deltan_j
                # iter.jacobian[(N-1) + (elements.size() + (j-gaseousSubstanceCount))*N] = H_D_j;
            # }
            # iter.jacobian[(N-2) + (N-2)*N] = -iter.n;
            # for (int j = 0; j < gaseousSubstanceCount; ++j) {
                # double H_D_j = substances[j]->GetH_D(iter.T);
                # for (auto &elem_i : substances[j]->composition) {
                    # // Molar balance equation partial derivatives of pi_i
                    # iter.jacobian[(N-2) + reverseElements[elem_i.first]*N] += elem_i.second*iter.n_j[j]; // a_ij*n_j

                    # // Enthalpy conservation equation partial derivatives of pi_i
                    # iter.jacobian[(N-1) + reverseElements[elem_i.first]*N] += elem_i.second*iter.n_j[j]*H_D_j; // a_ij*n_j*H_jstd/(R*T)
                # }
                # // Molar balance equation partial derivative of Deltaln_n
                # iter.jacobian[(N-2) + (N-2)*N] += iter.n_j[j];
                # // Molar balance equation partial derivative of Deltaln_T
                # iter.jacobian[(N-2) + (N-1)*N] += iter.n_j[j]*H_D_j; // n_j*H_jstd/(R*T)
            
                # // Enthalpy conservation equation partial derivative of Deltaln_n
                # iter.jacobian[(N-1) + (N-2)*N] += iter.n_j[j]*H_D_j; // n_j*H_jstd/(R*T)
                # // Enthalpy conservation equation partial derivative of Deltaln_T
                # iter.jacobian[(N-1) + (N-1)*N] += iter.n_j[j]*(substances[j]->GetCp_D(iter.T)+H_D_j*H_D_j); // (n_j*Cp_jstd/R+n_j*H_jstd^2/(R^2*T^2));
            # }

            # DGESV(&N, &nrhs, iter.jacobian, &N, iter.ipiv, iter.funcEvals, &N, &errorCode);
            # if (errorCode != 0) {
				# Log(Severity::ERROR, "Error solving matrix for chem eq step!");
                # result.valid = false;
				# return result;
            # }
            # for (int k = 0; k < elements.size(); ++k)
                # iter.pi_i[k] -= iter.funcEvals[k];
            # for (int j = 0; j < condensedSubstanceCount; ++j)
                # iter.Deltan_j[j] -= iter.funcEvals[j + elements.size()];
            # iter.Deltaln_n -= iter.funcEvals[N - 2];
            # iter.Deltaln_T -= iter.funcEvals[N - 1];

            # // Empirical lambda formulas suggested by NASA
            # double lambda1 = 5.0*std::max(abs(iter.Deltaln_T), abs(iter.Deltaln_n));
            # double lambda2 = HUGE_VAL;
            # for (int j = 0; j < gaseousSubstanceCount; ++j) {
                # double H_D_j = substances[j]->GetH_D(iter.T); // H_jstd/(R*T)
                # iter.Deltan_j_gas[j] = iter.Deltaln_n + H_D_j*iter.Deltaln_T -
                    # (H_D_j-substances[j]->GetS_D(iter.T)+log(iter.n_j[j]/iter.n)+log(iter.P/1.0E5));
                # for (auto& elem_i : substances[j]->composition) // sum across i, a_ij*pi_i
                    # iter.Deltan_j_gas[j] += elem_i.second*iter.pi_i[reverseElements[elem_i.first]];
                
                # double v = abs(iter.Deltan_j_gas[j]);
                # if (v > lambda1)
                    # lambda1 = v;
                # double ln_nj_n = log(iter.n_j[j]/iter.n);
                # if (ln_nj_n <= -18.420681 && iter.Deltan_j_gas[j] >= 0.0) {
                    # v = abs((-ln_nj_n-9.2103404) / (iter.Deltan_j_gas[j]-iter.Deltaln_n));
                    # if (v < lambda2)
                        # lambda2 = v;
                # }
            # }
            
            # // Limit corrections to prevent diverging solutions due to large jumps
            # lambda1 = 2.0 / lambda1;
            # double lambda = std::min({1.0, lambda1, lambda2});
            # for (int j = 0; j < gaseousSubstanceCount; ++j)
                # iter.n_j[j] *= exp(lambda*iter.Deltan_j_gas[j]);
            # for (int j = gaseousSubstanceCount; j < substances.size(); ++j)
                # iter.n_j[j] = iter.n_j[j]+lambda*iter.Deltan_j[j-gaseousSubstanceCount];
            # iter.n *= exp(lambda*iter.Deltaln_n);
            # iter.T *= exp(lambda*iter.Deltaln_T);

            # // Test for convergence, roughly testing the least expensive first
            # if (abs(iter.Deltaln_T) <= 1.0E-4) {
                # double sumN_j = 0.0;
                # for (int j = 0; j < substances.size(); ++j)
                    # sumN_j += iter.n_j[j];
                # double convergeGas = 0.0;
                # for (int j = 0; j < gaseousSubstanceCount; ++j)
                    # convergeGas += iter.n_j[j]*abs(iter.Deltan_j_gas[j]);
                # convergeGas /= sumN_j;
                # double convergeCondensed = 0.0;
                # for (int j = gaseousSubstanceCount; j < substances.size(); ++j)
                    # convergeCondensed += abs(iter.Deltan_j[j-gaseousSubstanceCount]);
                # convergeGas /= sumN_j;
                # double convergeTotal = iter.n*abs(iter.Deltaln_n)/sumN_j;
                # if (convergeGas <= 0.5E-5 && convergeCondensed <= 0.5E-5 && convergeTotal <= 0.5E-5) {
                    # bool massBalanceConvergence = true;
                    # for (int i = 0; i < elements.size(); ++i) {
                        # double sum_aij_nj = 0.0;
                        # for (const std::pair<int, double>& ja : relatedGasousSubstances[i])
                            # sum_aij_nj += ja.second * iter.n_j[ja.first];
                        # if (abs(iter.b_i0[i] - sum_aij_nj) > iter.b_i0_max*1.0E-6) {
                            # massBalanceConvergence = false;
                            # break;
                        # }
                    # }
                    # if (massBalanceConvergence)
                        # break;
                    # // TODO: Also do TRACE != 0 convergence test for pi_i
                # }
            # }
            # ++result.iterations;
        # }

        # result.T = iter.T;
        # result.M = 1/iter.n;
        # result.R = Universal::R / result.M;
        # //std::cerr << "result: " << result.T << " " << result.M << std::endl;

        # N = elements.size() + condensedSubstanceCount + 1;
        # // Can re-use as one less element is needed here and overriden data is useless after equilibrium iteration
        # double* tempDerivCoeffMatrix = iter.jacobian;
        # double* tempDerivs = iter.funcEvals;
        # double* pressDerivs = &iter.funcEvals[N];

        # for (int n = 0; n < N*N; ++n) tempDerivCoeffMatrix[n] = 0.0;
        # // dpi_i/dlnT then dn_j/dln_T then dlnn/dlnT
        # // Per-element equation
        # for (int k = 0; k < elements.size(); ++k) {
            # tempDerivs [k] = 0.0;
            # pressDerivs[k] = 0.0;
            # // TODO: Better way? > 1.0E-7 mainly prevents electron gas row from ruining the matrix
            # for (const std::pair<int, double> &ja : relatedGasousSubstances[k])
                # if (iter.n_j[ja.first] > 1.0E-7) {
                    # for (auto& elem_i : substances[ja.first]->composition) // a_kj*a_ij*n_j
                        # tempDerivCoeffMatrix[k + reverseElements[elem_i.first]*N] += ja.second*elem_i.second*iter.n_j[ja.first];
                    # tempDerivCoeffMatrix[k + (N-1)*N] += ja.second * iter.n_j[ja.first]; // a_kj*n_j
                    # tempDerivs [k] -= ja.second * iter.n_j[ja.first] * substances[ja.first]->GetH_D(iter.T); // a_kj*n_j*H_jstd/(R*T)
                    # pressDerivs[k] += ja.second * iter.n_j[ja.first]; // a_kj*n_j
                # }
            # for (const std::pair<int, double> &ja : relatedCondensedSubstances[k]) // a_kj
                # if (iter.n_j[ja.first] > 1.0E-7)
                    # tempDerivCoeffMatrix[k + (elements.size()+ja.first)*N] += ja.second;
            # // Basically check if there were any substances with this element with a concentration larger than 1.0E-7
            # for (int n = 0; n < N; ++n)
                # if (tempDerivCoeffMatrix[k + n*N] != 0.0)
                    # goto skip_edge_case;
            # // Just say that the derivative = 0.0
            # tempDerivCoeffMatrix[k + k*N] = 1.0;
            # tempDerivs [k] = 0.0; // TODO: Why was this != 0.0, very small? - may have forgotten to =0 above initially
            # pressDerivs[k] = 0.0;
        # skip_edge_case:;
        # }
        # // TODO: Something similar to above for very low concentrations?
        # // Per-condensed equation
        # for (int j = gaseousSubstanceCount; j < substances.size(); ++j) {
            # if (iter.n_j[j] > 1.0E-7) {
                # for (auto& elem_i : substances[j]->composition) // a_ij
                    # tempDerivCoeffMatrix[(elements.size()+(j-gaseousSubstanceCount))+(reverseElements[elem_i.first])*N] += elem_i.second;
                # tempDerivs[elements.size()+(j-gaseousSubstanceCount)] += -substances[j]->GetH_D(iter.T); // -H_jstd/(R*T)
            # }
            # pressDerivs[elements.size()+(j-gaseousSubstanceCount)] = 0.0;
        # }
        # // Singular equation
        # tempDerivs [N-1] = 0.0;
        # pressDerivs[N-1] = 0.0;
        # for (int j = 0; j < gaseousSubstanceCount; ++j)
            # if (iter.n_j[j] > 1.0E-7) {
                # for (auto &elem_i : substances[j]->composition) // a_ij*n_j
                    # tempDerivCoeffMatrix[(N-1)+reverseElements[elem_i.first]*N] += elem_i.second*iter.n_j[j];
                # tempDerivs [N-1] -= iter.n_j[j] * substances[j]->GetH_D(iter.T); // n_j*H_jstd/(R*T)
                # pressDerivs[N-1] += iter.n_j[j]; // n_j
            # }
        # nrhs = 2;
        # DGESV(&N, &nrhs, tempDerivCoeffMatrix, &N, iter.ipiv, tempDerivs, &N, &errorCode);
        # if (errorCode != 0) {
			# Log(Severity::ERROR, std::format("Error solving matrix for derivatives needed for equilibrium specific heat! LAPACK Code: {}", errorCode));
            # result.valid = false;
			# return result;
        # }
        # // TODO: If we are including condensed in Cp shouldn't they be in M too?
        # result.Cp = 0.0;
        # for (int j = 0; j < substances.size(); ++j) // Frozen contribution
            # result.Cp += iter.n_j[j]*substances[j]->GetCp_D(iter.T);
        # // TODO: Coeffs have already been computed in jacobian matrix, can we re-use?
        # for (int i = 0; i < elements.size(); ++i)
            # for (const std::pair<int, double>& ja : relatedGasousSubstances[i])
                # result.Cp += ja.second*iter.n_j[ja.first]*substances[ja.first]->GetH_D(iter.T)*tempDerivs[i];
        # for (int j = gaseousSubstanceCount; j < substances.size(); ++j)
            # result.Cp += substances[j]->GetH_D(iter.T)*tempDerivs[elements.size()+(j-gaseousSubstanceCount)];
        # for (int j = 0; j < gaseousSubstanceCount; ++j) {
            # double H_Dstd = substances[j]->GetH_D(iter.T);
            # result.Cp += iter.n_j[j]*(H_Dstd*tempDerivs[N-1]+H_Dstd*H_Dstd);
        # }
        # result.Cp *= Universal::R;

        # double dlnV_dlnT = tempDerivs [N-1] + 1.0; // dlnn/dlnT+1
        # double dlnV_dlnP = pressDerivs[N-1] - 1.0; // dlnn/dlnP-1
        # result.Cv      = result.Cp + result.R*dlnV_dlnT*dlnV_dlnT/dlnV_dlnP;
        # result.gamma   = result.Cp / result.Cv;
        # result.gamma_s = -result.gamma / dlnV_dlnP;

        # // TODO: Use gamma as that is standard for ideal gases
        # // gamma_s, isentropic expansion factor, is not defined by the usual relationship for a reacting mixture
        # //     Due to more compilcated derivatives, see https://en.wikipedia.org/wiki/Relations_between_heat_capacities
        # // Cp-Cv=R, k=Cp/Cv, Cv=Cp-R, k=Cp/(Cp-R), k=1/(1-R/Cp)
        # //result.k = 1.0/(1.0-result.R/result.Cp);
        # if (returnComposition)
            # for (int j = 0; j < substances.size(); ++j)
                # result.composition.push_back(std::pair<ThermochemicalSubstance*, double>(substances[j], iter.n_j[j]*result.M));

        # return result;
    # }

    # /*
    # Unimplented NASA Recommendations:
    # Only compute n_j from lnn_j from mole fractions larger than a threshold (during update) otherwise set to 0 (max 35 iterations)
        # There are separate thresholds for ionized and non ionized species. Ratios for overly abundant species
    # Use P_0 (for gibbs energy) relevant to the specific source (use 1bar for all right now)
    # Singularity detection and identification



    # Differences between other versions:
    # Introduced a limitation where species significantly outside of the available temperature ranges are not considered
        # Need to move out of initialization and change initial guess for these to 0 and leave out unless it becomes relevant
    # Doesn't support condensed. Nasa supports condensed by first solving then adding condensed and resuming.
        # Current, at start, condensed species want to become negative. Could just try taking the abs of the n_j
    # */
# }

