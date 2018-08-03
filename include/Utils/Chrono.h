#include <chrono>

namespace Utils {

class Chrono
{
public:
	Chrono(Chrono const&) = delete;
	void operator=(Chrono const&) = delete;

	using TimeTy = std::chrono::time_point<std::chrono::steady_clock>;

	static TimeTy Start();
	static long End(TimeTy start);

private:
	Chrono() {}

	static Chrono& GetInstance()
	{
		static Chrono instance;
		return instance;
	}
};

}
