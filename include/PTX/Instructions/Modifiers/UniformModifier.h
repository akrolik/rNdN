#pragma once

namespace PTX {

class UniformModifier
{
public:
	UniformModifier(bool uniform = false) : m_uniform(uniform) {}

	bool GetUniform() const { return m_uniform; }
	void SetUniform(bool uniform) { m_uniform = uniform; }

protected:
	bool m_uniform = false;
};

}
