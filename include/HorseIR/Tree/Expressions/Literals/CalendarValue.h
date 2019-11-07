#pragma once

namespace HorseIR {

class CalendarValue
{
public:
	virtual std::int32_t GetEpochTime() const = 0;
};

}
