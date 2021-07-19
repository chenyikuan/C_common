#ifndef CYK_SINGLETON_H_
#define CYK_SINGLETON_H_

#include <iostream>
#include <list>
#include <algorithm>

// using namespace std;

class CSingletonBase
{
protected:
	//////////////////////////////////////////////////////
	//nested class
	class CInstanceTable : public std::list<CSingletonBase *>
	{
	public:
		CInstanceTable()
		{
			m_bIsClearing = false;
		};

		virtual ~CInstanceTable()
		{
			m_bIsClearing = true;
			for_each(begin(), end(), DeleteInstance);
		}

	public:
		static void DeleteInstance(CSingletonBase * pInstance)
		{
			delete pInstance;
		}

	public:
		bool m_bIsClearing;
	};
	//end of nested class
	///////////////////////////////////////////////////////////

public:
	CSingletonBase()
	{
		CSingletonBase::m_InstanceTbl.push_back(this);
	}

	virtual ~CSingletonBase()
	{
		if(!m_InstanceTbl.m_bIsClearing)
			m_InstanceTbl.remove(this);
	}

public:
	//static member
	static CInstanceTable m_InstanceTbl;
};


template <typename T>
class cyk_singleton : public CSingletonBase
{
public:
	// double check lock pattern.
	static T * get_instance()
	{
		if(!m_pInstance)
		{
			//这里需要使用同步机制！！！
			if(!m_pInstance) //must check again!
			{
				m_pInstance = new T();
			}
		}

		return m_pInstance;
	}

protected:
	// 所有继承该模板的类，均需设置friend class cyk_singleton<cykTools>;作为友元类，并将自身的构造函数设置为私有
	cyk_singleton()
	{
		//std::cout << "------- Singleton created! -------" << std::endl;
	}

	virtual ~cyk_singleton()
	{
		m_pInstance = NULL;
		//std::cout << "------- Singleton destroyed! -------" << std::endl;
	}

private:
	static T *m_pInstance;
};

//must be defined here.
//if defined in singleton.cpp there will be a link error.
template<typename T> T * cyk_singleton<T>::m_pInstance = NULL;

/*
usage:
	class cyk_test : public cyk_singleton<cyk_test>
	{
	protected://or private:  this will prevent programmer creating object directly
	    friend class cyk_singleton<cyk_test>;
	    cyk_test(){};
	public:
		virtual ~cyk_test() {};
	    int a;
	};

OR simple:
	class cyk_test : public cyk_singleton<cyk_test>
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
