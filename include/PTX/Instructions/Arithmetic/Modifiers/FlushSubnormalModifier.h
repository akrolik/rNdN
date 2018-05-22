#pragma once

namespace PTX
{

class FlushSubnormalModifier
{
public:
	FlushSubnormalModifier() {}
	FlushSubnormalModifier(bool flush) : m_flush(flush) {}

	bool GetFlushSubnormal() const { return m_flush; }
	void SetFlushSubnormal(bool flush) { m_flush = flush; }

protected:
	bool m_flush = false;
};

}
