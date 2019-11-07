#pragma once

namespace HorseIR {

class ExtendedCalendarValue
{
public:
	virtual double GetExtendedEpochTime() const = 0;
};

}
