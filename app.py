from flask import Flask, render_template, request, jsonify, session
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import Counter
import time
import html
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Last.fm API configuration
API_KEY = 'ac92a37f42f55eee4a5fcd7321e752fe'
API_SECRET = '051d35a38d4f1bfada0be98cf806ce60'
API_BASE_URL = 'http://ws.audioscrobbler.com/2.0/'

# Request retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Global variables to store user data
track_features_map = {}
tag_A = {}
tag_B = {}

# 增加 genre 主类归一化映射
GENRE_MAIN_CLASS = {
    'pop': 'pop',
    'dance': 'pop',
    'synthpop': 'pop',
    'electropop': 'pop',
    'k-pop': 'pop',
    'j-pop': 'pop',
    'rock': 'rock',
    'classic rock': 'rock',
    'hard rock': 'rock',
    'metal': 'rock',
    'punk': 'rock',
    'alternative rock': 'rock',
    'indie rock': 'rock',
    'jazz': 'jazz',
    'blues': 'jazz',
    'hip hop': 'hiphop',
    'hiphop': 'hiphop',
    'rap': 'hiphop',
    'trap': 'hiphop',
    'classical': 'classical',
    'orchestra': 'classical',
    'folk': 'folk',
    'country': 'folk',
    'electronic': 'electronic',
    'edm': 'electronic',
    'techno': 'electronic',
    'house': 'electronic',
    'trance': 'electronic',
    'dubstep': 'electronic',
    'r&b': 'rnb',
    'rnb': 'rnb',
    'soul': 'rnb',
    'funk': 'rnb',
    'reggae': 'reggae',
    'latin': 'latin',
    'salsa': 'latin',
    'flamenco': 'latin',
    'indie': 'indie',
    'alternative': 'alternative',
}

# 添加时代映射
ERA_MAPPING = {
    '50s': '1950s',
    '60s': '1960s',
    '70s': '1970s',
    '80s': '1980s',
    '90s': '1990s',
    '2000s': '2000s',
    '2010s': '2010s',
    '2020s': '2020s',
}

def safe_api_call(method, params, retries=MAX_RETRIES):
    """Safely call the Last.fm API with error handling and retries"""
    params['api_key'] = API_KEY
    params['format'] = 'json'
    params['method'] = method
    
    for attempt in range(retries):
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                logger.warning(f"API call failed, retrying ({attempt+1}/{retries}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"API call failed after multiple retries: {e}")
                raise
        except Exception as e:
            logger.error(f"Other API error: {e}")
            raise

def is_api_available():
    """Check if Last.fm API is available"""
    try:
        # Try a simple API call to check connection
        result = safe_api_call('chart.getTopArtists', {'limit': 1})
        return True
    except Exception as e:
        logger.error(f"API unavailable: {e}")
        return False

def get_track_info(track_name, artist_name=None):
    """Get track information from Last.fm API"""
    params = {'track': track_name}
    if artist_name:
        params['artist'] = artist_name
        
    try:
        result = safe_api_call('track.getInfo', params)
        return result
    except Exception as e:
        logger.error(f"Error getting track info: {e}")
        return None

def search_track(track_name):
    """Search for tracks on Last.fm"""
    try:
        result = safe_api_call('track.search', {'track': track_name, 'limit': 5})
        
        if 'results' in result and 'trackmatches' in result['results'] and 'track' in result['results']['trackmatches']:
            tracks = result['results']['trackmatches']['track']
            # Last.fm might return a single dict for one result, convert to list
            if isinstance(tracks, dict):
                tracks = [tracks]
            return tracks
        return []
    except Exception as e:
        logger.error(f"Error searching track: {e}")
        return []

def get_track_tags(track_name, artist_name):
    """Get tags for a track from Last.fm API"""
    try:
        result = safe_api_call('track.getTopTags', {'track': track_name, 'artist': artist_name})
        
        if 'toptags' in result and 'tag' in result['toptags']:
            tags = result['toptags']['tag']
            # Last.fm might return a single dict for one tag, convert to list
            if isinstance(tags, dict):
                tags = [tags]
            return tags
        return []
    except Exception as e:
        logger.error(f"Error getting track tags: {e}")
        return []

def get_similar_tracks(track_name, artist_name):
    """Get similar tracks from Last.fm API"""
    try:
        result = safe_api_call('track.getSimilar', {'track': track_name, 'artist': artist_name, 'limit': 10})
        
        if 'similartracks' in result and 'track' in result['similartracks']:
            similar_tracks = result['similartracks']['track']
            # Last.fm might return a single dict for one result, convert to list
            if isinstance(similar_tracks, dict):
                similar_tracks = [similar_tracks]
            return similar_tracks
        return []
    except Exception as e:
        logger.error(f"Error getting similar tracks: {e}")
        return []

def extract_track_features(track_info):
    """Extract and normalize musical features from track information and tags"""
    if not track_info:
        return None
    
    # Get track tags
    artist_name = track_info.get('artist', {}).get('name', '')
    track_name = track_info.get('name', '')
    
    tags = get_track_tags(track_name, artist_name)
    
    # 调试信息：打印获取到的标签
    logger.info(f"处理歌曲: {track_name} - {artist_name}")
    logger.info(f"获取到的标签数量: {len(tags)}")
    for tag in tags[:10]:  # 打印前10个标签
        logger.info(f"  - {tag.get('name', '')}")
    
    # Initialize features with default values
    features = {
        'danceability': 0.5,
        'energy': 0.5,
        'acousticness': 0.5,
        'instrumentalness': 0.5,
        'liveness': 0.5,
        'valence': 0.5,
        'speechiness': 0.5,
        'tempo': 120,  # Default tempo
        'genre': 'unknown',
        'language': 'unknown',
        'era': 'unknown',  # 添加时代特征
    }
    
    # Map common tags to musical features, genre, and language
    tag_mapping = {
        # 舞曲特性
        'dance': ('danceability', 0.8),
        'danceable': ('danceability', 0.8),
        'edm': ('danceability', 0.9),
        'club': ('danceability', 0.8),
        'party': ('danceability', 0.7),
        
        # 能量特性
        'energetic': ('energy', 0.9),
        'high energy': ('energy', 0.9),
        'powerful': ('energy', 0.8),
        'intense': ('energy', 0.9),
        'calm': ('energy', 0.2),
        'chill': ('energy', 0.3),
        'relaxing': ('energy', 0.2),
        'mellow': ('energy', 0.3),
        'soft': ('energy', 0.2),
        
        # 原声特性
        'acoustic': ('acousticness', 0.9),
        'unplugged': ('acousticness', 0.8),
        'live acoustic': ('acousticness', 0.9),
        
        # 器乐特性
        'instrumental': ('instrumentalness', 0.9),
        'no vocals': ('instrumentalness', 0.9),
        'orchestra': ('instrumentalness', 0.8),
        
        # 现场特性
        'live': ('liveness', 0.9),
        'concert': ('liveness', 0.8),
        'recorded live': ('liveness', 0.9),
        
        # 情感特性
        'happy': ('valence', 0.9),
        'sad': ('valence', 0.2),
        'melancholy': ('valence', 0.3),
        'uplifting': ('valence', 0.8),
        'dark': ('valence', 0.2),
        'emotional': ('valence', 0.4),
        'upbeat': ('valence', 0.8),
        
        # 语音特性
        'rap': ('speechiness', 0.9),
        'spoken word': ('speechiness', 0.9),
        'talking': ('speechiness', 0.8),
        'vocals': ('speechiness', 0.7),
        
        # 速度特性
        'fast': ('tempo', 160),
        'slow': ('tempo', 80),
        'medium tempo': ('tempo', 120),
        
        # 流派
        'pop': ('genre', 'pop'),
        'rock': ('genre', 'rock'),
        'jazz': ('genre', 'jazz'),
        'blues': ('genre', 'blues'),
        'hip hop': ('genre', 'hiphop'),
        'hip-hop': ('genre', 'hiphop'),
        'rap': ('genre', 'hiphop'),
        'classical': ('genre', 'classical'),
        'metal': ('genre', 'metal'),
        'folk': ('genre', 'folk'),
        'country': ('genre', 'country'),
        'electronic': ('genre', 'electronic'),
        'edm': ('genre', 'electronic'),
        'techno': ('genre', 'electronic'),
        'house': ('genre', 'electronic'),
        'trance': ('genre', 'electronic'),
        'dubstep': ('genre', 'electronic'),
        'r&b': ('genre', 'rnb'),
        'rnb': ('genre', 'rnb'),
        'soul': ('genre', 'rnb'),
        'funk': ('genre', 'rnb'),
        'reggae': ('genre', 'reggae'),
        'latin': ('genre', 'latin'),
        'salsa': ('genre', 'latin'),
        'flamenco': ('genre', 'latin'),
        'indie': ('genre', 'indie'),
        'alternative': ('genre', 'alternative'),
        'punk': ('genre', 'punk'),
        'grunge': ('genre', 'grunge'),
        'disco': ('genre', 'disco'),
        'k-pop': ('genre', 'k-pop'),
        'j-pop': ('genre', 'j-pop'),
        'c-pop': ('genre', 'c-pop'),
        'mandopop': ('genre', 'c-pop'),
        'cantopop': ('genre', 'c-pop'),
        
        # 语言
        'english': ('language', 'english'),
        'chinese': ('language', 'chinese'),
        'mandarin': ('language', 'chinese'),
        'cantonese': ('language', 'chinese'),
        'japanese': ('language', 'japanese'),
        'french': ('language', 'french'),
        'german': ('language', 'german'),
        'spanish': ('language', 'spanish'),
        'korean': ('language', 'korean'),
        'italian': ('language', 'italian'),
        'russian': ('language', 'russian'),
        'portuguese': ('language', 'portuguese'),
        'hindi': ('language', 'hindi'),
        'arabic': ('language', 'arabic'),
        'turkish': ('language', 'turkish'),
        'thai': ('language', 'thai'),
        'vietnamese': ('language', 'vietnamese'),
        
        # 时代
        '50s': ('era', '1950s'),
        '60s': ('era', '1960s'),
        '70s': ('era', '1970s'),
        '80s': ('era', '1980s'),
        '90s': ('era', '1990s'),
        '2000s': ('era', '2000s'),
        '2010s': ('era', '2010s'),
        '2020s': ('era', '2020s'),
        'oldies': ('era', 'oldies'),
        'classic': ('era', 'classic'),
        'retro': ('era', 'retro'),
        'vintage': ('era', 'vintage'),
        'modern': ('era', 'modern'),
        'contemporary': ('era', 'contemporary'),
    }
    
    # Extract tag names and count
    tag_count = 0
    if tags:
        for tag in tags:
            tag_name = tag.get('name', '').lower()
            tag_count += 1
            # Map tag to feature/genre/language
            for keyword, (feature, value) in tag_mapping.items():
                if keyword in tag_name:
                    features[feature] = value
    
    # Include track listeners and playcount as popularity metrics
    features['popularity'] = float(track_info.get('listeners', 0)) / 1000000  # Normalize by million
    
    # If duration is available, convert to seconds
    if 'duration' in track_info:
        try:
            features['duration'] = int(track_info['duration']) / 1000  # Convert ms to seconds
        except (ValueError, TypeError):
            features['duration'] = 0
    
    # 归一化 genre 到主类
    genre = features.get('genre', 'unknown')
    features['genre_main'] = GENRE_MAIN_CLASS.get(genre, genre)
    
    # 保存原始标签列表用于重叠度判定
    features['tags'] = tags
    
    # 调试信息：打印提取的特征
    logger.info("提取的特征:")
    for key, value in features.items():
        if key != 'tags':  # 不打印完整标签列表
            logger.info(f"  - {key}: {value}")
    
    return features

def analyze_music_features(features_list):
    """Analyze a group of tracks' features and generate tags"""
    if not features_list:
        return {}
    
    # Calculate averages
    avg_features = {}
    for key in ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'speechiness', 'tempo', 'popularity']:
        values = [features[key] for features in features_list if key in features]
        if values:
            avg_features[key] = sum(values) / len(values)
    
    # Generate descriptive tags
    tags = {}
    
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
    
    if avg_features['popularity'] > 0.5:
        tags['popularity'] = 'mainstream'
    elif avg_features['popularity'] < 0.1:
        tags['popularity'] = 'niche'
    else:
        tags['popularity'] = 'moderate'
    
    # 添加主要流派和语言
    genre_counter = Counter()
    language_counter = Counter()
    era_counter = Counter()
    
    for features in features_list:
        if 'genre_main' in features and features['genre_main'] != 'unknown':
            genre_counter[features['genre_main']] += 1
        if 'language' in features and features['language'] != 'unknown':
            language_counter[features['language']] += 1
        if 'era' in features and features['era'] != 'unknown':
            era_counter[features['era']] += 1
    
    # 添加最常见的流派、语言和时代
    if genre_counter:
        tags['genre_main'] = genre_counter.most_common(1)[0][0]
    if language_counter:
        tags['language'] = language_counter.most_common(1)[0][0]
    if era_counter:
        tags['era'] = era_counter.most_common(1)[0][0]
    
    # Add average features
    tags['avg_features'] = avg_features
    
    return tags

def filter_dissimilar_songs(reference_songs, candidate_songs):
    """Filter tracks that are significantly different from the reference group"""
    if not reference_songs or not candidate_songs:
        return []
    
    # Get reference group feature vectors
    reference_features = []
    reference_genre_main = None
    reference_genre = None
    reference_language = None
    reference_era = None
    reference_artist = None
    reference_tags = None
    
    for song in reference_songs:
        if song in track_features_map:
            features = track_features_map[song]['features']
            feature_vector = [
                features['danceability'], features['energy'], 
                features['speechiness'], features['acousticness'], 
                features['instrumentalness'], features['liveness'], 
                features['valence']
            ]
            reference_features.append(feature_vector)
            
            # 收集参考组的特征
            if reference_genre_main is None:
                reference_genre_main = features.get('genre_main', 'unknown')
            if reference_genre is None:
                reference_genre = features.get('genre', 'unknown')
            if reference_language is None:
                reference_language = features.get('language', 'unknown')
            if reference_era is None:
                reference_era = features.get('era', 'unknown')
            if reference_artist is None:
                reference_artist = track_features_map[song].get('artist', None)
            if reference_tags is None:
                reference_tags = set()
                if 'tags' in features and isinstance(features['tags'], list):
                    reference_tags = set([t['name'].lower() for t in features['tags'] if 'name' in t])
    
    if not reference_features:
        return []
    
    # 收集参考组的关键标签类别
    reference_tag_categories = set()
    important_categories = ['rock', 'pop', 'jazz', 'classical', 'hiphop', 'electronic', 'metal', 
                           'folk', 'rnb', 'reggae', 'latin', 'indie', 'alternative']
    
    for tag in reference_tags:
        for category in important_categories:
            if category in tag:
                reference_tag_categories.add(category)
                break
    
    # Calculate reference group average vector
    reference_avg = np.mean(reference_features, axis=0)
    
    # 加权向量 - 增加能量和情感的权重
    weights = np.array([1.2, 1.5, 0.8, 0.8, 0.8, 0.8, 1.5])
    
    # Calculate similarity for each candidate song
    dissimilar_songs = []
    for song in candidate_songs:
        if song in track_features_map:
            features = track_features_map[song]['features']
            reason = None
            
            # 1. 语言不同
            if features.get('language', 'unknown') != reference_language and reference_language != 'unknown' and features.get('language', 'unknown') != 'unknown':
                reason = f"语言不同: {features.get('language', 'unknown')} vs {reference_language}"
                dissimilar_songs.append({"id": song, "reason": reason})
                continue
            
            # 2. 时代不同
            if features.get('era', 'unknown') != reference_era and reference_era != 'unknown' and features.get('era', 'unknown') != 'unknown':
                reason = f"时代不同: {features.get('era', 'unknown')} vs {reference_era}"
                dissimilar_songs.append({"id": song, "reason": reason})
                continue
            
            # 3. 主风格不同
            if features.get('genre_main', 'unknown') != reference_genre_main and reference_genre_main != 'unknown' and features.get('genre_main', 'unknown') != 'unknown':
                reason = f"主风格不同: {features.get('genre_main', 'unknown')} vs {reference_genre_main}"
                dissimilar_songs.append({"id": song, "reason": reason})
                continue
                
            # 4. 子风格不同
            if features.get('genre', 'unknown') != reference_genre and reference_genre != 'unknown' and features.get('genre', 'unknown') != 'unknown':
                reason = f"子风格不同: {features.get('genre', 'unknown')} vs {reference_genre}"
                dissimilar_songs.append({"id": song, "reason": reason})
                continue
                
            # 5. 标签重叠度判定 - 降低阈值
            song_tags = set()
            if 'tags' in features and isinstance(features['tags'], list):
                song_tags = set([t['name'].lower() for t in features['tags'] if 'name' in t])
            
            if reference_tags and song_tags:
                overlap = len(reference_tags & song_tags) / max(1, len(reference_tags | song_tags))
                if overlap < 0.2:  # 从0.3降低到0.2
                    reason = f"标签重叠度低: {overlap:.2f}"
                    dissimilar_songs.append({"id": song, "reason": reason})
                    continue
            
            # 6. 检查关键标签类别
            song_tag_categories = set()
            for tag in song_tags:
                for category in important_categories:
                    if category in tag:
                        song_tag_categories.add(category)
                        break
            
            category_diff = reference_tag_categories.symmetric_difference(song_tag_categories)
            if len(category_diff) > 1:
                reason = f"关键标签类别不同: {', '.join(category_diff)}"
                dissimilar_songs.append({"id": song, "reason": reason})
                continue
                    
            # 7. 歌手优先判定
            song_artist = track_features_map[song].get('artist', None)
            if song_artist == reference_artist:
                threshold = 2.0
            else:
                threshold = 1.5
                
            # 8. 加权距离
            feature_vector = [
                features['danceability'], features['energy'], 
                features['speechiness'], features['acousticness'], 
                features['instrumentalness'], features['liveness'], 
                features['valence']
            ]
            distance = np.linalg.norm((np.array(feature_vector) - reference_avg) * weights)
            if distance > threshold:
                reason = f"音频特征差异大: 距离值 {distance:.2f} > {threshold}"
                dissimilar_songs.append({"id": song, "reason": reason})
    
    return dissimilar_songs

def compare_tags(tag_a, tag_b):
    """Compare differences between two tag sets"""
    comparison = {}
    
    # Compare basic tags
    for key in ['energy', 'danceability', 'tempo', 'style', 'mood', 'popularity', 'genre_main', 'language', 'era']:
        if key in tag_a and key in tag_b:
            comparison[key] = {
                'tag_a': tag_a[key],
                'tag_b': tag_b[key],
                'different': tag_a[key] != tag_b[key]
            }
    
    # Compare average feature values
    if 'avg_features' in tag_a and 'avg_features' in tag_b:
        feature_diffs = {}
        for feature, value_a in tag_a['avg_features'].items():
            if feature in tag_b['avg_features']:
                diff = abs(value_a - tag_b['avg_features'][feature])
                feature_diffs[feature] = {
                    'tag_a': round(value_a, 2),
                    'tag_b': round(tag_b['avg_features'][feature], 2),
                    'diff': round(diff, 2),
                    'significant': diff > 0.2  # Consider difference significant if > 0.2
                }
        comparison['feature_diffs'] = feature_diffs
    
    return comparison

@app.route('/')
def index():
    """Serve the single page application"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    """Return API availability status"""
    status = is_api_available()
    return jsonify({'status': 'available' if status else 'unavailable'})

@app.route('/api/search_track', methods=['POST'])
def api_search_track():
    """Search for tracks and return results"""
    query = request.json.get('query', '')
    if not query:
        return jsonify({'success': False, 'error': 'No search query provided'})
    
    try:
        tracks = search_track(query)
        results = []

        # 先筛选歌名完全等于query的（忽略大小写）
        exact_matches = [
            track for track in tracks
            if track.get('name', '').strip().lower() == query.strip().lower()
        ]
        # 如果有完全匹配，优先返回这些
        filtered_tracks = exact_matches if exact_matches else tracks

        for track in filtered_tracks[:5]:
            track_name = track.get('name', '')
            artist_name = track.get('artist', '')
            results.append({
                'name': track_name,
                'artist': artist_name,
                'id': f"{artist_name}:{track_name}"  # Create a unique ID
            })
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logger.error(f"搜索歌曲时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process_track', methods=['POST'])
def api_process_track():
    """Process a track and extract its features"""
    track_name = request.json.get('track', '')
    artist_name = request.json.get('artist', '')
    song_key = request.json.get('id', '')
    
    if not track_name or not artist_name:
        return jsonify({'success': False, 'error': 'Track and artist required'})
    
    try:
        # Get detailed track info
        track_info = get_track_info(track_name, artist_name)
        
        if not track_info or 'track' not in track_info:
            return jsonify({'success': False, 'error': 'Track not found'})
        
        # Extract features
        features = extract_track_features(track_info['track'])
        
        if not features:
            return jsonify({'success': False, 'error': 'Failed to extract features'})
        
        # Store in global map
        track_features_map[song_key] = {
            'name': track_name,
            'artist': artist_name,
            'features': features
        }
        
        return jsonify({
            'success': True, 
            'name': track_name,
            'artist': artist_name,
            'features': features
        })
    except Exception as e:
        logger.error(f"处理歌曲时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze_group', methods=['POST'])
def api_analyze_group():
    """Analyze a group of tracks and generate tags"""
    group_id = request.json.get('group_id', '')
    song_keys = request.json.get('songs', [])
    
    if not song_keys:
        return jsonify({'success': False, 'error': 'No songs provided'})
    
    try:
        # Get features for valid songs
        valid_features = [track_features_map[key]['features'] for key in song_keys if key in track_features_map]
        
        if not valid_features:
            return jsonify({'success': False, 'error': 'No valid songs found'})
        
        # Generate tags
        tags = analyze_music_features(valid_features)
        
        # Store tags globally
        if group_id == 'A':
            global tag_A
            tag_A = tags
        elif group_id == 'B':
            global tag_B
            tag_B = tags
        
        return jsonify({'success': True, 'tags': tags})
    except Exception as e:
        logger.error(f"分析歌曲组时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/filter_songs', methods=['POST'])
def api_filter_songs():
    """Filter dissimilar songs between two groups"""
    group_a_songs = request.json.get('group_a', [])
    group_b_songs = request.json.get('group_b', [])
    
    if not group_a_songs or not group_b_songs:
        return jsonify({'success': False, 'error': 'Both song groups required'})
    
    try:
        # Filter dissimilar songs
        filtered_songs = filter_dissimilar_songs(group_a_songs, group_b_songs)
        
        # Get detailed info for filtered songs
        filtered_details = []
        for song_item in filtered_songs:
            song_key = song_item["id"]
            if song_key in track_features_map:
                filtered_details.append({
                    'id': song_key,
                    'name': track_features_map[song_key]['name'],
                    'artist': track_features_map[song_key]['artist'],
                    'reason': song_item["reason"]
                })
        
        return jsonify({
            'success': True, 
            'filtered_songs': filtered_details,
            'total_count': len(group_b_songs),
            'filtered_count': len(filtered_songs)
        })
    except Exception as e:
        logger.error(f"筛选歌曲时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/compare_groups', methods=['POST'])
def api_compare_groups():
    """Compare two groups of songs and their tags"""
    try:
        # Compare tag_A and tag_B
        comparison = compare_tags(tag_A, tag_B)
        
        return jsonify({
            'success': True,
            'tag_a': tag_A,
            'tag_b': tag_B,
            'comparison': comparison
        })
    except Exception as e:
        logger.error(f"比较歌曲组时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)