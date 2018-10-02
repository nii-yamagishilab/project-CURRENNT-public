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

#include "JsonClasses.hpp"

#include <stdexcept>
#include <string>


namespace helpers {

    int safeJsonGetInt(const JsonValue &val, const char *name)
    {
        if (val->HasMember(name))
            return (*val)[name].GetInt();
        else
            return 0;
    }

    template <>
    bool checkedJsonGet<bool>(const JsonDocument &jsonDoc, const char *varName)
    {
        if (!jsonDoc->HasMember(varName) || !(*jsonDoc)[varName].IsBool())
            throw std::runtime_error(std::string("Variable '") + varName + "' is missing or has the wrong type");

        return (*jsonDoc)[varName].GetBool();
    }

    template <>
    int checkedJsonGet<int>(const JsonDocument &jsonDoc, const char *varName)
    {
        if (!jsonDoc->HasMember(varName) || !(*jsonDoc)[varName].IsInt())
            throw std::runtime_error(std::string("Variable '") + varName + "' is missing or has the wrong type");

        return (*jsonDoc)[varName].GetInt();
    }

    template <>
    double checkedJsonGet<double>(const JsonDocument &jsonDoc, const char *varName)
    {
        if (!jsonDoc->HasMember(varName) || (!(*jsonDoc)[varName].IsDouble() && !(*jsonDoc)[varName].IsInt()))
            throw std::runtime_error(std::string("Variable '") + varName + "' is missing or has the wrong type");

        if ((*jsonDoc)[varName].IsDouble())
            return (*jsonDoc)[varName].GetDouble();
        else
            return (double)(*jsonDoc)[varName].GetInt();
    }

    template <>
    float checkedJsonGet<float>(const JsonDocument &jsonDoc, const char *varName)
    {
        return (float)checkedJsonGet<double>(jsonDoc, varName);
    }

} // namespace helpers
