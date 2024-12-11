#pragma once
#include <string>
#include <vector>
#include <map>
#include <stack>

using std::string;
using std::map;
using std::vector;

#define MAX_THING (1 << 25)

class TempData{
private:
	static const int size = 60;
	map<string, int> M;
	map<string, void*> Mlarge;
	bool* used;
	int* mem;
	int* count;
public:
	TempData();
	int* useCount();
	int getCount();
	template<typename T>
	T* use(const string& name, int num = 1)
	{
		if (M.find(name) == M.end())
		{
			int l = (sizeof(T) * num) / 4;
			bool flag;
			for (int i = 0; i <= size - l; i++)
			{
				flag = true;
				for(int j=0; j<l; j++)
					if (used[i + j] == true) { flag = false; break;}
				if (flag)
				{
					M[name] = i;
					for (int j = 0; j < l; j++)
						used[i + j] = true;
					return (T*)(mem + (i * MAX_THING));
				}
			}
			printf("%s: no mem\n", name);
			return nullptr;
		}
		else {
			printf("%s: name reuse\n", name);
			return nullptr;
		}
	}
	
	template<typename T>
	T* useLarge(const string& name, int size)
	{
		if (Mlarge.find(name) == Mlarge.end())
		{
			T* temp = nullptr;
			cudaMalloc((void**)&temp, size * sizeof(T));
			Mlarge[name] = temp;
			return temp;
		}
		else {
			printf("%s: name reuse\n", name);
			return nullptr;
		}
	}

	template<typename T>
	T* unuseLarge(const string& name)
	{
		auto it = Mlarge.find(name);
		if (it == Mlarge.end())
		{
			printf("%s: unuse not find\n", name);
		}
		else {
			cudaFree(it->second);
		}
	}

	template<typename T>
	void unuse(const string& name, int num = 1)
	{
		
		auto it = M.find(name);
		if (it == M.end())
		{
			printf("%s: unuse not find\n", name);
		}
		else {
			int l = (sizeof(T) * num) / 4;
			for (int j = 0; j < l; j++)
			{
				used[it->second + j] = false;
			}
			M.erase(it);
		}
	}

	template<typename T>
	T* get(const string& name)
	{
		auto it = M.find(name);
		if (it == M.end())
		{
			printf("error5\n");
		}
		else {
			return (T*)(mem+(it->second * MAX_THING));
		}
	}

	template<typename T>
	T* getLarge(const string& name)
	{
		auto it = Mlarge.find(name);
		if (it == Mlarge.end())
		{
			printf("error6\n");
		}
		else {
			return (T*)it->second;
		}
	}

	~TempData();
};
