#pragma once

#include <ostream>

namespace HorseIR {

class ComplexValue
{
public:
	ComplexValue(double real, double imag) : m_real(real), m_imaginary(imag) {}

	double GetReal() const { return m_real; }
	void SetReal(double real) { m_real = real; }

	double GetImaginary() const { return m_imaginary; }
	void SetImaginary(double imag) { m_imaginary = imag; }

	friend std::ostream& operator<<(std::ostream& os, const ComplexValue& value);

	bool operator==(const ComplexValue& other) const
	{
		return (m_real == other.m_real && m_imaginary == other.m_imaginary);
	}

	bool operator!=(const ComplexValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	double m_real = 0.0;
	double m_imaginary = 0.0;
};

inline std::ostream& operator<<(std::ostream& os, const ComplexValue& value)
{
	os << value.m_real;
	if (value.m_imaginary >= 0)
	{
		os << "+";
	}
	os << value.m_imaginary << "i";
	return os;
}

}
