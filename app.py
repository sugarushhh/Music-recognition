from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import Counter

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Spotify API 配置
CLIENT_ID = '3395bd6dd71448e599805be8255c2437'
CLIENT_SECRET = '66431b32b0b04f078991a1486b5b9eb5'
REDIRECT_URI = 'https://two61272-s-project.onrender.com
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# 保存用户数据的全局变量
audio_features_map = {}
tag_A = {}
tag_B = {}

@app.route('/')
def index():
    return render_template('step1.html')

@app.route('/step1', methods=['POST'])
def step1():
    if request.method == 'POST':
        # 获取用户输入的歌曲名单
        songs_text = request.form.get('songs')
        songs_list = [song.strip() for song in songs_text.split('\n') if song.strip()]
        
        # 获取歌曲特征
        results = []
        for song in songs_list:
            try:
                # 搜索歌曲
                search_results = sp.search(q=song, type='track', limit=1)
                if search_results['tracks']['items']:
                    track = search_results['tracks']['items'][0]
                    track_id = track['id']
                    
                    # 获取音频特征
                    audio_features = sp.audio_features(track_id)[0]
                    
                    # 保存到全局映射
                    if audio_features:
                        audio_features_map[song] = {
                            'id': track_id,
                            'name': track['name'],
                            'artists': ', '.join([artist['name'] for artist in track['artists']]),
                            'features': audio_features
                        }
                        
                        results.append({
                            'name': track['name'],
                            'artists': ', '.join([artist['name'] for artist in track['artists']]),
                            'success': True
                        })
                    else:
                        results.append({'name': song, 'success': False, 'error': '无法获取音频特征'})
                else:
                    results.append({'name': song, 'success': False, 'error': '找不到歌曲'})
            except Exception as e:
                results.append({'name': song, 'success': False, 'error': str(e)})
        
        # 生成tag_A
        global tag_A
        tag_A = analyze_music_features([audio_features_map[song]['features'] for song in songs_list if song in audio_features_map])
        
        session['songs_list_1'] = songs_list
        session['results_1'] = results
        
        return render_template('step2.html', results=results)

@app.route('/step2', methods=['POST'])
def step2():
    if request.method == 'POST':
        # 获取用户输入的第二组歌曲名单
        songs_text = request.form.get('songs')
        songs_list = [song.strip() for song in songs_text.split('\n') if song.strip()]
        
        # 获取歌曲特征
        results = []
        for song in songs_list:
            try:
                # 搜索歌曲
                search_results = sp.search(q=song, type='track', limit=1)
                if search_results['tracks']['items']:
                    track = search_results['tracks']['items'][0]
                    track_id = track['id']
                    
                    # 获取音频特征
                    audio_features = sp.audio_features(track_id)[0]
                    
                    # 保存到全局映射
                    if audio_features:
                        audio_features_map[song] = {
                            'id': track_id,
                            'name': track['name'],
                            'artists': ', '.join([artist['name'] for artist in track['artists']]),
                            'features': audio_features
                        }
                        
                        results.append({
                            'name': track['name'],
                            'artists': ', '.join([artist['name'] for artist in track['artists']]),
                            'success': True
                        })
                    else:
                        results.append({'name': song, 'success': False, 'error': '无法获取音频特征'})
                else:
                    results.append({'name': song, 'success': False, 'error': '找不到歌曲'})
            except Exception as e:
                results.append({'name': song, 'success': False, 'error': str(e)})
        
        session['songs_list_2'] = songs_list
        session['results_2'] = results
        
        # 筛选差异较大的歌曲
        filtered_songs = filter_dissimilar_songs(session['songs_list_1'], songs_list)
        session['filtered_songs'] = filtered_songs
        
        return render_template('review.html', 
                               results=results, 
                               filtered_songs=filtered_songs, 
                               total_songs=len(songs_list),
                               filtered_count=len(filtered_songs))

@app.route('/accept_results', methods=['POST'])
def accept_results():
    # 用户接受筛选结果
    filtered_songs = session.get('filtered_songs', [])
    
    # 生成tag_B
    global tag_B
    tag_B = analyze_music_features([audio_features_map[song]['features'] for song in filtered_songs if song in audio_features_map])
    
    # 比较tag_A和tag_B
    comparison = compare_tags(tag_A, tag_B)
    
    return render_template('result.html', 
                           filtered_songs=filtered_songs,
                           tag_a=tag_A,
                           tag_b=tag_B,
                           comparison=comparison)

@app.route('/reject_results', methods=['POST'])
def reject_results():
    # 用户拒绝筛选结果，返回第二步
    return render_template('step2.html', results=session.get('results_2', []))

def analyze_music_features(features_list):
    """分析一组歌曲的音频特征，生成标签"""
    if not features_list:
        return {}
    
    # 提取主要特征
    feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                    'speechiness', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'tempo']
    
    # 计算平均值
    avg_features = {}
    for key in feature_keys:
        values = [features[key] for features in features_list if key in features]
        if values:
            avg_features[key] = sum(values) / len(values)
    
    # 确定音乐风格特征
    tags = {}
    
    # 根据平均特征生成描述性标签
    if avg_features['energy'] > 0.7:
        tags['energy'] = 'high'
    elif avg_features['energy'] < 0.4:
        tags['energy'] = 'low'
    else:
        tags['energy'] = 'medium'
        
    if avg_features['danceability'] > 0.7:
        tags['danceability'] = 'high'
    elif avg_features['danceability'] < 0.4:
        tags['danceability'] = 'low'
    else:
        tags['danceability'] = 'medium'
    
    if avg_features['tempo'] > 120:
        tags['tempo'] = 'fast'
    elif avg_features['tempo'] < 90:
        tags['tempo'] = 'slow'
    else:
        tags['tempo'] = 'medium'
    
    if avg_features['acousticness'] > 0.7:
        tags['style'] = 'acoustic'
    elif avg_features['instrumentalness'] > 0.5:
        tags['style'] = 'instrumental'
    elif avg_features['speechiness'] > 0.4:
        tags['style'] = 'speech-heavy'
    
    if avg_features['valence'] > 0.7:
        tags['mood'] = 'positive'
    elif avg_features['valence'] < 0.3:
        tags['mood'] = 'negative'
    else:
        tags['mood'] = 'neutral'
    
    # 添加其他特征
    tags['avg_features'] = avg_features
    
    return tags

def filter_dissimilar_songs(reference_songs, candidate_songs):
    """筛选出与参考歌曲组特征差异较大的歌曲"""
    if not reference_songs or not candidate_songs:
        return []
    
    # 获取参考组的特征向量
    reference_features = []
    for song in reference_songs:
        if song in audio_features_map:
            features = audio_features_map[song]['features']
            feature_vector = [
                features['danceability'], features['energy'], 
                features['speechiness'], features['acousticness'], 
                features['instrumentalness'], features['liveness'], 
                features['valence']
            ]
            reference_features.append(feature_vector)
    
    if not reference_features:
        return []
    
    # 计算参考组的平均特征向量
    reference_avg = np.mean(reference_features, axis=0)
    
    # 计算每个候选歌曲与参考组的相似度
    dissimilar_songs = []
    for song in candidate_songs:
        if song in audio_features_map:
            features = audio_features_map[song]['features']
            feature_vector = [
                features['danceability'], features['energy'], 
                features['speechiness'], features['acousticness'], 
                features['instrumentalness'], features['liveness'], 
                features['valence']
            ]
            
            # 计算欧氏距离
            distance = np.linalg.norm(np.array(feature_vector) - reference_avg)
            
            # 如果距离大于阈值，认为是差异较大的歌曲
            if distance > 0.8:
                dissimilar_songs.append(song)
    
    return dissimilar_songs

def compare_tags(tag_a, tag_b):
    """比较两组标签的差异"""
    comparison = {}
    
    # 比较基本标签
    for key in ['energy', 'danceability', 'tempo', 'style', 'mood']:
        if key in tag_a and key in tag_b:
            comparison[key] = {
                'tag_a': tag_a[key],
                'tag_b': tag_b[key],
                'different': tag_a[key] != tag_b[key]
            }
    
    # 比较平均特征值
    if 'avg_features' in tag_a and 'avg_features' in tag_b:
        feature_diffs = {}
        for feature, value_a in tag_a['avg_features'].items():
            if feature in tag_b['avg_features']:
                diff = abs(value_a - tag_b['avg_features'][feature])
                feature_diffs[feature] = {
                    'tag_a': round(value_a, 2),
                    'tag_b': round(tag_b['avg_features'][feature], 2),
                    'diff': round(diff, 2),
                    'significant': diff > 0.2  # 如果差异大于0.2，认为是显著差异
                }
        comparison['feature_diffs'] = feature_diffs
    
    return comparison

if __name__ == '__main__':
    app.run(debug=True)