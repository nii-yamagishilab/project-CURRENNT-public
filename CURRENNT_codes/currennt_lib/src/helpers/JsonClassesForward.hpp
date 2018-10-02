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

#ifndef HELPERS_JSONCLASSESFORWARD_HPP
#define HELPERS_JSONCLASSESFORWARD_HPP


namespace helpers {

    class JsonValue;
    class JsonAllocator;
    class JsonDocument;

    int safeJsonGetInt(const JsonValue &val, const char *name);
    
    template <typename T>
    T checkedJsonGet(const JsonDocument &jsonDoc, const char *varName);

} // namespace helpers


#endif // HELPERS_JSONCLASSESFORWARD_HPP
