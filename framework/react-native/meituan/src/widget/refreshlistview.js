import React, { Component } from 'react'
import { StyleSheet, TouchableOpacity, ListView, RefreshControl, View, Text, ActivityIndicator } from 'react-native'

export const RefreshState = {
  Idle: 'Idle',
  Refreshing: 'Refreshing',
  NoMoreData: 'NoMoreData',
  Failure: 'Failure'
}

class RefreshListView extends Component {
  static propTypes = {
    onHeaderRefresh: React.PropTypes.func,
    onFooterRefresh: React.PropTypes.func
  }

  static defaultProps = {
    footerRefreshingText: '数据加载中……',
    footerFailureText: '点击重新加载',
    footerNoMoreDataText: '已加载全部数据'
  }

  constructor(props: Object) {
    super(props)
    this.state = {
      headerState: RefreshState.Idle,
      footerState: RefreshState.Idle
    }
  }

  render() {
    return (
      <ListView {...this.props}
        enableEmptySections
        refreshControl={
          <RefreshControl
            refreshing={this.state.headerState == RefreshState.Refreshing}
            onRefresh={() => this.onHeaderRefresh()}
            tintColor='gray'
          />
        }
        renderFooter={() => this.renderFooter()}
        onEndReachedThreshold={10}
        onEndReached={() => this.onFooterRefresh()}
      />
    )
  }

  renderFooter() {
    return (
      <RefreshListFooter
        onPress={() => this.startFooterRefreshing()}
        footerRefreshingText={this.props.footerRefreshingText}
        footerNoMoreDataText={this.props.footerNoMoreDataText}
      />
    )
  }

  headState() {
    return self.state.footerState
  }
  shouldStartHeaderRefreshing() {
    if (this.state.headerState == RefreshState.Refreshing ||
        this.state.footerState == RefreshState.Refreshing) {
      return false
    }

    return true
  }
  startHeaderRefreshing() {
    this.setState({headerState: RefreshState.Refreshing})

    if (this.props.onHeaderRefresh) {
      this.props.onHeaderRefresh()
    }
  }
  onHeaderRefresh() {
    if (this.shouldStartHeaderRefreshing()) {
      this.startHeaderRefreshing()
    }
  }

  footerState() {
    return self.state.footerState
  }
  shouldStartFooterRefreshing() {
    if (this.state.headerState == RefreshState.Refreshing ||
        this.state.footerState == RefreshState.Refreshing) {
      return false
    }
    if (this.state.footerState == RefreshState.Failure ||
        this.state.footerState == RefreshState.NoMoreData) {
      return false
    }
    if (this.props.dataSource.getRowCount() == 0) {
      return false
    }
    return true
  }
  startFooterRefreshing() {
    this.setState({footerState: RefreshState.Refreshing})

    if (this.props.onFooterRefresh) {
      this.props.onFooterRefresh()
    }
  }
  onFooterRefresh() {
    if (this.shouldStartFooterRefreshing()) {
      this.startFooterRefreshing()
    }
  }

  endRefreshing(refreshState: RefreshState) {
    if (refreshState == RefreshState.Refreshing) {
      return
    }

    let footerState = refreshState
    if (this.props.dataSource.getRowCount() == 0) {
      footerState = RefreshState.Idle
    }

    this.setState({
      headerState: RefreshState.Idle,
      footerState: footerState
    })
  }
}

class RefreshListFooter extends Component {
  render() {
    let onPress = this.props.onPress
    let refreshingText = this.props.footerRefreshingText
    let noMoreDataText = this.props.footerNoMoreDataText

    let footer = null
    switch (this.state.footerState) {
      case RefreshState.Idle: {
        break;
      }
      case RefreshState.Failure: {
        footer = (
          <TouchableOpacity style={styles.footerContainer} onPress={onPress}>
            <Text1 style={styles.footerText}>{this.props.footerFailureText}</Text1>
          </TouchableOpacity>
        )
        break;
      }
      case RefreshState.Refreshing: {
        footer = (
          <View style={styles.footerContainer}>
            <ActivityIndicator size="small" color="#888888" />
            <Text style={styles.footerText}>
              {refreshingText}
            </Text>
          </View>
        )
        break;
      }
      case RefreshState.NoMoreData: {
        footer = (
          <View style={styles.footerContainer}>
            <Text style={styles.footerText}>
              {noMoreDataText}
            </Text>
          </View>
        )
        break;
      }
    }

    return footer
  }
}

const styles = StyleSheet.create({
  footerContainer: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 10,
  },
  footerText: {
    footSize: 14,
    color: '#555555',
  }
})

export default RefreshListView