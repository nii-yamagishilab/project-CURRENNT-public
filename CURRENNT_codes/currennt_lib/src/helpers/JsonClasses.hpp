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

#ifndef HELPERS_JSONCLASSES_HPP
#define HELPERS_JSONCLASSES_HPP

#include "JsonClassesForward.hpp"
#include "../rapidjson/document.h"


namespace helpers {

    class JsonValue
    {
    private:
        mutable rapidjson::Value *_p;

    public:
        JsonValue()
            : _p(NULL)
        {
        }

        JsonValue(rapidjson::Value *p)
            : _p(p)
        {
        }

        bool isValid() const
        {
            return (_p != NULL);
        }

        rapidjson::Value* operator-> () const
        {
            return _p;
        }

        rapidjson::Value& operator* () const
        {
            return *_p;
        }
    };

    class JsonAllocator
    {
    private:
        mutable rapidjson::MemoryPoolAllocator<> *_p;

    public:
        JsonAllocator(rapidjson::MemoryPoolAllocator<> *p)
            : _p(p)
        {
        }

        rapidjson::MemoryPoolAllocator<>* operator-> () const
        {
            return _p;
        }

        operator rapidjson::MemoryPoolAllocator<>& () const
        {
            return *_p;
        }
    };

    class JsonDocument
    {
    private:
        mutable rapidjson::Document *_p;

    public:
        JsonDocument(rapidjson::Document *p)
            : _p(p)
        {
        }

        JsonDocument(rapidjson::Document &p)
            : _p(&p)
        {
        }

        rapidjson::Document* operator-> () const
        {
            return _p;
        }

        rapidjson::Document& operator* () const
        {
            return *_p;
        }

        operator rapidjson::Document& () const
        {
            return *_p;
        }
    };

} // namespace helpers


#endif // HELPERS_JSONCLASSES_HPP
