#include <chrono>

namespace Utils {

class Chrono
{
public:
	Chrono(Chrono const&) = delete;
	void operator=(Chrono const&) = delete;

	using TimeTy = std::chrono::time_point<std::chrono::steady_clock>;

	static TimeTy Start()
	{
		return std::chrono::steady_clock::now();
	}

	static long Finish(TimeTy start)
	{
		auto end = std::chrono::steady_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}

private:
	Chrono() {}

	static Chrono& GetInstance()
	{
		static Chrono instance;
		return instance;
	}
};

}
