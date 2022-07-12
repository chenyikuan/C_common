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

/*
usage:
    class cyk_test : public CSingleton<cyk_test>
    {
    protected://or private:  this will prevent programmer creating object directly
        friend class CSingleton<cyk_test>;
        cyk_test(){};
    public:
        virtual ~cyk_test() {};
        int a;
    };

OR simple:
    class cyk_test : public CSingleton<cyk_test>
    {
    public:
        int a;
    };

Call:
    cyk_test* x1 = cyk_test::get_instance();
    cyk_test* x2 = cyk_test::get_instance();
    x1->a = 10;
    cout << x2->a << endl;
*/

#endif  //CYK_SINGLETON_H_
