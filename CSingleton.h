#ifndef CYK_SINGLETON_H_
#define CYK_SINGLETON_H_

#include <iostream>
#include <list>
#include <algorithm>
#include <mutex>
#include <memory>

template <typename T>
class CSingleton {
public:
	static T* get_instance() {
		static std::once_flag onceFlag; // 必须是静态的
		std::call_once(onceFlag, [&] {m_instance = new T(); }); // 只会调用一次
		return m_instance;
	};
	
protected:
	CSingleton() {}; //私有构造函数，不允许使用者自己生成对象，但是必须要实现
	CSingleton(const CSingleton& other) = delete;
	CSingleton& operator = (const CSingleton& other) = delete;
    ~CSingleton() {delete m_instance;};

private:
	static T* m_instance; //静态成员变量 
};

template<typename T> T* CSingleton<T>::m_instance = nullptr;

#endif  //CYK_SINGLETON_H_
