#include"Median.h"

#define IS_EVEN _set.size()%2==0
#define IS_ODD _set.size()%2>0


template <class ItemType>
typename Median<ItemType>::iterator Median<ItemType>::insert(ItemType item)
{
    if(_set.empty())
    {
        mIter=_set.insert(item);
        return mIter;
    }
    iterator iter;
    if(item<*mIter)
    {
        iterator iter=_set.insert(item);
        if(IS_EVEN)
            mIter--;
        return iter;
    }
    else
    {
        iterator iter=_set.insert(mIter,item);
        if(IS_ODD)
            mIter++;
        return iter;
    }
}

template <class ItemType>
void Median<ItemType>::remove(iterator iter)
{
    if(iter!=mIter)
    {
        ItemType item=*iter;
        _set.erase(iter);
        if(item<*mIter)
        {
            if(IS_ODD)
                mIter++;
        }
        else
        {
            if(IS_EVEN)
                mIter--;
        }
    }
    else
    {
        if(IS_ODD)
             mIter--;
        else
             mIter++;
        _set.erase(iter);
    }
}

template <class ItemType>
ItemType Median<ItemType>::median() const
{
    if(_set.empty())
        return ItemType();
    if(IS_ODD)
    {
        return *mIter;
    }
    else
    {
        iterator tmpIter=mIter;
        tmpIter++;
        ItemType ret= *mIter + *tmpIter;
        return ret/2;
    }
}

template <class ItemType>
std::size_t Median<ItemType>::size() const
{
    return _set.size();
}

template <class ItemType>
void Median<ItemType>::clear()
{
    return _set.clear();
    mIter=_set.end();
}

template <class ItemType>
Median<ItemType>::~Median()
{
    _set.clear();
}

//============WindowMedian=============
template <class ItemType>
WindowMedian<ItemType>::WindowMedian(std::size_t windowSize)
    :_windowSize(windowSize)
{
}


template <class ItemType>
std::size_t WindowMedian<ItemType>::window() const
{
    return _windowSize;
}

template <class ItemType>
void WindowMedian<ItemType>::insert(ItemType item)
{
    typename Median<ItemType>::iterator inserted=med.insert(item);
    if(size()>_windowSize)
    {
        typename Median<ItemType>::iterator oldest=olderQ.front();
        med.remove(oldest);
        olderQ.pop();
    }
    olderQ.push(inserted);
}

template <class ItemType>
ItemType WindowMedian<ItemType>::median() const
{
    return med.median();
}

template <class ItemType>
void WindowMedian<ItemType>::clear()
{
    med.clear();
    while (!olderQ.empty())
    {
        olderQ.pop();
    }
}

template <class ItemType>
std::size_t WindowMedian<ItemType>::size() const
{
    return med.size();
}
