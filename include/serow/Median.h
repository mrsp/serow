#ifndef MEDIAN_H
#define MEDIAN_H

#include <set>
#include <queue>
template <class ItemType>
class Median
{
    public:
        typedef typename std::multiset<ItemType>::iterator iterator;
        Median(){}
        ~Median();
        iterator insert(ItemType item);
        void remove(iterator iter);
        ItemType median() const;

        inline std::size_t size() const;
        inline void clear();

    protected:
        std::multiset<ItemType> _set;

        //mIter always points at the median element in odd sized sets.
        //in even sizes mIter points at the lower element
        iterator mIter;
};

template <class ItemType>
class WindowMedian
{
    public:
//        typedef typename std::multiset<ItemType>::iterator iterator;
        WindowMedian(std::size_t windowSize);
        inline std::size_t window() const;
        void insert(ItemType item);
        inline ItemType median() const;
        inline std::size_t size() const;
        inline void clear();
    private:
        std::size_t _windowSize;
        std::queue<typename Median<ItemType>::iterator> olderQ;
        Median<ItemType> med;

};

#include <serow/Median.tpp>
#endif
