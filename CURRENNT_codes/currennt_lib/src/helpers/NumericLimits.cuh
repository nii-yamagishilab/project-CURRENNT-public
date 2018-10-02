/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef HELPERS_NUMERICLIMITS_CUH
#define HELPERS_NUMERICLIMITS_CUH

#define RETURN_VALUE_FN(fnName, value) static inline __host__ __device__ value_type fnName() { return value; }


namespace helpers {

    template <typename T>
    struct NumericLimits;

    template <>
    struct NumericLimits<float>
    {
        typedef float value_type;

        RETURN_VALUE_FN(min,      1.1754944e-038f);
        RETURN_VALUE_FN(max,      3.4028235e+038f);
        RETURN_VALUE_FN(expLimit, 88.722839f);
        RETURN_VALUE_FN(logInf,   1e30f);
        RETURN_VALUE_FN(logZero,  -1e30f);
	// Add numeric support (borrowed from HTS)
	RETURN_VALUE_FN(lSMALL,   -0.5e10f);
	RETURN_VALUE_FN(minLarg,  1.1754944e-038f); //  = exp(-87.3365)
	RETURN_VALUE_FN(minEarg,  -87.3365);        //  = log(1.1754944e-038)
	RETURN_VALUE_FN(minLogExp,-69.0776);        //  = -log(-logZero)
    };

    template <>
    struct NumericLimits<double>
    {
        typedef double value_type;

        RETURN_VALUE_FN(min,      2.2250738585072014e-308);
        RETURN_VALUE_FN(max,      1.7976931348623157e+308);
        RETURN_VALUE_FN(expLimit, 709.78271289338397);
        RETURN_VALUE_FN(logInf,   1e100);
        RETURN_VALUE_FN(logZero,  -1e100);
	// NOTE: not changed for double
	RETURN_VALUE_FN(lSMALL,   -0.5e10f);
	RETURN_VALUE_FN(minLarg,  1.1754944e-038f); //  = exp(-87.3365)
	RETURN_VALUE_FN(minEarg,  -87.3365);        //  = log(1.1754944e-038)
	RETURN_VALUE_FN(minLogExp,-69.0776);        //  = -log(-logZero)
    };

} // namespace helpers


#endif // HELPERS_NUMERICLIMITS_CUH
