#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class FlushSubnormalModifier
{
};

template<class T, bool force>
class FlushSubnormalModifier<T, force, std::enable_if_t<force || T::FlushModifier>>
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
