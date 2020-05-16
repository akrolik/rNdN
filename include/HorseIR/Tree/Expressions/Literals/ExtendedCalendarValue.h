#pragma once

namespace HorseIR {

class ExtendedCalendarValue
{
public:
	virtual std::int64_t GetExtendedEpochTime() const = 0;
};

}
